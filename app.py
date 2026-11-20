import gc, os, shutil, tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import librosa
from PIL import Image, ImageDraw, ImageFont
from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, CompositeVideoClip, ImageClip
from moviepy.video.fx.all import crop
from moviepy.audio.fx.all import audio_loop
from nudenet import NudeDetector
import whisper

st.set_page_config(page_title="YouTube Shorts Maker", page_icon="✂️", layout="wide")
st.title("✂️ YouTube Shorts Maker")
st.caption("Make 30-second vertical clips from videos you have permission to reuse.")

@st.cache_resource
def load_nudity_detector():
    return NudeDetector()

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

def download_video(url, folder):
    # Prefer separate video/audio streams and merge; fall back to progressive formats.
    opts={
        "format":"bv*[height<=720]+ba/b[height<=720]/b",
        "outtmpl":str(Path(folder)/"source.%(ext)s"),
        "merge_output_format":"mp4",
        "noplaylist":True,
        "quiet":True,
        "no_warnings":True,
        "retries":3,
        "fragment_retries":3,
    }
    with YoutubeDL(opts) as ydl:
        info=ydl.extract_info(url,download=True)
        expected=ydl.prepare_filename(info)
    # Account for postprocessor changing the extension after stream merge.
    candidates=[p for p in Path(folder).glob("source.*") if p.suffix.lower() in {".mp4",".mkv",".webm",".mov"}]
    if not candidates:
        raise RuntimeError("Download finished but no video file was found. This video may have unavailable formats.")
    path=max(candidates,key=lambda p:p.stat().st_size)
    if path.stat().st_size<=0:
        raise RuntimeError("Downloaded video file is empty.")
    return str(path),info

def energy_segments(video_path,count,duration=30):
    audio_path=str(Path(video_path).with_suffix(".analysis.wav"))
    with VideoFileClip(video_path) as clip:
        if clip.audio is None: raise RuntimeError("Source video has no audio track for energy-based selection.")
        total=clip.duration
        clip.audio.write_audiofile(audio_path,fps=16000,nbytes=2,codec="pcm_s16le",logger=None)
    try:
        y,sr=librosa.load(audio_path,sr=16000,mono=True)
    finally:
        try: os.unlink(audio_path)
        except OSError: pass
    if total<=duration: return [0.0]
    candidates=np.arange(0,max(0,total-duration)+0.01,5.0)
    scores=[]
    for start in candidates:
        seg=y[int(start*sr):int((start+duration)*sr)]
        if len(seg)<sr*duration*0.8:
            scores.append(0.0); continue
        rms=librosa.feature.rms(y=seg,frame_length=2048,hop_length=512)[0]
        scores.append(float(np.mean(rms)))
    ranked=sorted(zip(candidates,scores),key=lambda x:x[1],reverse=True)
    chosen=[]
    for start,_score in ranked:
        if all(abs(float(start)-prev)>=duration*0.75 for prev in chosen):
            chosen.append(float(start))
            if len(chosen)>=count: break
    return sorted(chosen) or [0.0]

def has_exposed_content(clip,detector,threshold=0.45,folder=None):
    exposed={"FEMALE_BREAST_EXPOSED","FEMALE_GENITALIA_EXPOSED","MALE_GENITALIA_EXPOSED","BUTTOCKS_EXPOSED"}
    scan_dir=folder or tempfile.gettempdir()
    scan_path=Path(scan_dir)/"nudenet_scan_frame.jpg"
    for t in np.arange(0,max(clip.duration,0.01),3.0):
        frame=clip.get_frame(float(min(t,clip.duration-0.001)))
        Image.fromarray(frame).save(scan_path,quality=85)
        try:
            detections=detector.detect(str(scan_path))
            if any(d.get("class") in exposed and float(d.get("score",0))>threshold for d in detections):
                return True
        finally:
            try: scan_path.unlink()
            except OSError: pass
    return False

def crop_vertical(clip):
    w,h=clip.size
    target=9/16
    if w/h>target:
        return crop(clip,width=int(h*target),height=h,x_center=w/2,y_center=h/2).resize((1080,1920))
    return crop(clip,width=w,height=int(w/target),x_center=w/2,y_center=h/2).resize((1080,1920))

def make_caption_image(text,width=1080):
    canvas=Image.new("RGBA",(width,240),(0,0,0,0)); draw=ImageDraw.Draw(canvas)
    try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",48)
    except OSError: font=ImageFont.load_default()
    lines=[]; line=""
    for word in text.split():
        candidate=(line+" "+word).strip()
        if draw.textbbox((0,0),candidate,font=font)[2]>width-100 and line:
            lines.append(line); line=word
        else: line=candidate
    if line: lines.append(line)
    lines=lines[-3:]; y=canvas.height-(len(lines)*58)-24
    for line in lines:
        box=draw.textbbox((0,0),line,font=font,stroke_width=3)
        x=(width-(box[2]-box[0]))//2
        draw.text((x,y),line,font=font,fill="white",stroke_width=4,stroke_fill="black")
        y+=58
    return np.array(canvas)

def add_captions(video,model,temp_dir):
    if video.audio is None: return video
    audio_path=str(Path(temp_dir)/"caption_audio.wav")
    video.audio.write_audiofile(audio_path,fps=16000,nbytes=2,codec="pcm_s16le",logger=None)
    try: transcript=model.transcribe(audio_path,fp16=False)
    finally:
        try: os.unlink(audio_path)
        except OSError: pass
    overlays=[]
    for seg in transcript.get("segments",[]):
        start=max(0,float(seg.get("start",0))); end=min(video.duration,float(seg.get("end",0)))
        text=seg.get("text","").strip()
        if not text or end<=start: continue
        overlay=ImageClip(make_caption_image(text)).set_start(start).set_duration(end-start).set_position(("center","bottom"))
        overlays.append(overlay)
    return CompositeVideoClip([video]+overlays).set_duration(video.duration) if overlays else video

def build_short(source,start,voice_path,music_path,source_vol,voice_vol,music_vol,captions,model,detector,work):
    source_clip=VideoFileClip(source)
    base=None; vertical=None; final=None; voice=None; music=None
    try:
        end=min(start+30,source_clip.duration)
        base=source_clip.subclip(start,end)
        vertical=crop_vertical(base)
        if vertical.duration<30:
            vertical=vertical.loop(duration=30)
        vertical=vertical.set_duration(30)
        if has_exposed_content(vertical,detector,folder=work):
            return None,"Automated exposure screening flagged this clip."
        layers=[]
        if vertical.audio is not None: layers.append(vertical.audio.volumex(source_vol))
        if voice_path:
            voice=AudioFileClip(voice_path)
            voice=voice.subclip(0,min(30,voice.duration)) if voice.duration>30 else voice
            layers.append(voice.volumex(voice_vol))
        if music_path:
            music=AudioFileClip(music_path)
            music=audio_loop(music,duration=30).set_duration(30)
            layers.append(music.volumex(music_vol))
        if layers: vertical=vertical.set_audio(CompositeAudioClip(layers).set_duration(30))
        final=vertical
        if captions: final=add_captions(final,model,work)
        return final,None
    except Exception:
        if final and final is not vertical: final.close()
        raise
    finally:
        source_clip.close()
        if base: base.close()
        if voice: voice.close()
        if music: music.close()

st.warning("Use only content you own or are licensed to reuse. Automated exposure screening can be inaccurate and is not a substitute for moderation.")
url=st.text_input("YouTube video URL")
count=st.slider("Number of clips",1,5,3)
source_vol=st.slider("Original audio volume",0.0,2.0,1.0,0.1)
voice_vol=st.slider("Voiceover volume",0.0,2.0,1.0,0.1)
music_vol=st.slider("Background music volume",0.0,1.0,0.25,0.05)
voice_file=st.file_uploader("Optional commentary/voiceover",type=["mp3","wav"])
music_file=st.file_uploader("Optional background audio",type=["mp3","wav"])
captions=st.checkbox("Burn automatic captions (Whisper tiny)",True)

if st.button("Generate Shorts",type="primary",disabled=not url):
    work=tempfile.mkdtemp(prefix="shortmaker_")
    try:
        with st.spinner("Downloading source video up to 720p…"):
            source,info=download_video(url,work)
        st.write(f"**Source:** {info.get('title','Video')}")
        starts=energy_segments(source,count)
        detector=load_nudity_detector()
        model=load_whisper() if captions else None
        voice_path=music_path=None
        if voice_file:
            voice_path=os.path.join(work,"voice."+voice_file.name.rsplit(".",1)[-1].lower())
            with open(voice_path,"wb") as f: f.write(voice_file.getvalue())
        if music_file:
            music_path=os.path.join(work,"music."+music_file.name.rsplit(".",1)[-1].lower())
            with open(music_path,"wb") as f: f.write(music_file.getvalue())
        rendered=0
        for idx,start in enumerate(starts,1):
            with st.spinner(f"Processing selected segment {idx}/{len(starts)}…"):
                video,err=build_short(source,start,voice_path,music_path,source_vol,voice_vol,music_vol,captions,model,detector,work)
                if err:
                    st.warning(f"Segment {idx}: {err}"); continue
                output=os.path.join(work,f"short_{idx}.mp4")
                try:
                    video.write_videofile(output,codec="libx264",audio_codec="aac",fps=30,threads=2,preset="ultrafast",logger=None)
                    with open(output,"rb") as f: data=f.read()
                    st.video(data)
                    st.download_button(f"Download Short {idx}",data,file_name=f"short_{idx}.mp4",mime="video/mp4",key=f"download_{idx}")
                    rendered+=1
                finally:
                    video.close(); gc.collect()
        if not rendered: st.error("No clips were successfully rendered.")
    except Exception as e:
        st.error(f"Processing error: {e}")
        st.info("Try fewer clips or a shorter source video. Processing may exceed cloud resource limits.")
    finally:
        shutil.rmtree(work,ignore_errors=True)
