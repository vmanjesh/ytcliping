import os, tempfile, shutil, gc, subprocess, math
from pathlib import Path
import streamlit as st
import numpy as np
import librosa
from PIL import Image, ImageDraw, ImageFont
from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, ImageClip
from moviepy.video.fx.all import crop
from nudenet import NudeDetector
import whisper

st.set_page_config(page_title="YouTube Shorts Maker", page_icon="✂️", layout="wide")
st.title("✂️ YouTube Shorts Maker")
st.caption("Create 30-second vertical clips from videos you have permission to use.")

@st.cache_resource
def load_nudity_detector():
    return NudeDetector()

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

def download_video(url, folder):
    opts = {"format":"best[height<=720]/best","outtmpl":str(Path(folder)/"source.%(ext)s"),"noplaylist":True,"quiet":True,"no_warnings":True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    return path, info

def energy_segments(video_path, count, duration=30):
    audio_path = str(Path(video_path).with_suffix(".wav"))
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec="pcm_s16le", logger=None)
        total = clip.duration
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    os.unlink(audio_path)
    if total <= duration:
        return [0.0]
    hop = 16000
    window = 30 * sr
    candidates = np.arange(0, max(1, int(total-duration)), 5.0)
    scores = []
    for start in candidates:
        s=int(start*sr); e=min(len(y), s+window)
        seg=y[s:e]
        if len(seg) < sr*duration*0.8: scores.append(0); continue
        rms=librosa.feature.rms(y=seg, frame_length=2048, hop_length=hop)[0]
        scores.append(float(np.mean(rms)))
    ranked=sorted(zip(candidates,scores),key=lambda x:x[1],reverse=True)
    chosen=[]
    for start,score in ranked:
        if all(abs(start-prev)>=duration*0.65 for prev in chosen):
            chosen.append(float(start))
            if len(chosen)>=count: break
    return sorted(chosen) or [0.0]

def has_exposed_content(video, detector, threshold=0.45):
    classes={"FEMALE_BREAST_EXPOSED","FEMALE_GENITALIA_EXPOSED","MALE_GENITALIA_EXPOSED","BUTTOCKS_EXPOSED"}
    fps=max(1, int(video.fps or 24))
    for t in np.arange(0, video.duration, 3):
        frame=video.get_frame(float(t))
        img=Image.fromarray(frame)
        tmp=Path(tempfile.gettempdir())/("nudenet_scan.jpg")
        img.save(tmp, quality=80)
        try:
            detections=detector.detect(str(tmp))
            if any(d.get("class") in classes and d.get("score",0)>threshold for d in detections):
                return True
        finally:
            try: tmp.unlink()
            except: pass
    return False

def crop_vertical(clip):
    w,h=clip.size
    target=9/16
    if w/h>target:
        nw=int(h*target)
        return crop(clip, width=nw, height=h, x_center=w/2, y_center=h/2).resize((1080,1920))
    nh=int(w/target)
    return crop(clip, width=w, height=nh, x_center=w/2, y_center=h/2).resize((1080,1920))

def make_caption_image(text, width=1080):
    canvas=Image.new("RGBA",(width,240),(0,0,0,0))
    draw=ImageDraw.Draw(canvas)
    try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",48)
    except: font=ImageFont.load_default()
    words=text.split(); lines=[]; line=""
    for word in words:
        test=(line+" "+word).strip()
        if draw.textbbox((0,0),test,font=font)[2]>width-100 and line:
            lines.append(line); line=word
        else: line=test
    if line: lines.append(line)
    lines=lines[-3:]
    y=canvas.height-(len(lines)*58)-24
    for line in lines:
        box=draw.textbbox((0,0),line,font=font,stroke_width=3)
        x=(width-(box[2]-box[0]))//2
        draw.text((x,y),line,font=font,fill="white",stroke_width=4,stroke_fill="black")
        y+=58
    return np.array(canvas)

def caption_clip_for_segment(model, segment, offset, duration):
    audio=segment.audio
    if audio is None: return segment
    temp_audio=Path(tempfile.gettempdir())/f"short_audio_{os.getpid()}.wav"
    audio.write_audiofile(str(temp_audio),fps=16000,nbytes=2,codec="pcm_s16le",logger=None)
    try:
        result=model.transcribe(str(temp_audio),fp16=False)
    finally:
        try: temp_audio.unlink()
        except: pass
    overlays=[]
    for item in result.get("segments",[]):
        start=max(0,float(item["start"])); end=min(duration,float(item["end"]))
        if end<=start: continue
        img=make_caption_image(item.get("text","").strip())
        overlays.append(ImageClip(img,transparent=True).set_start(start).set_duration(end-start).set_position(("center","bottom")))
    return CompositeVideoClip([segment]+overlays).set_duration(duration) if overlays else segment

def build_short(source, start, voice_path, music_path, original_volume, voice_volume, music_volume, captions, model, detector):
    source_clip=VideoFileClip(source)
    raw=source_clip.subclip(start,min(start+30,source_clip.duration))
    vertical=crop_vertical(raw)
    if vertical.duration<30:
        vertical=vertical.loop(duration=30)
    vertical=vertical.set_duration(30)
    if has_exposed_content(vertical, detector):
        source_clip.close(); raw.close(); vertical.close()
        return None, "Rejected: exposed-content scan triggered."
    audio_layers=[]
    if vertical.audio is not None:
        audio_layers.append(vertical.audio.volumex(original_volume))
    extras=[]
    if voice_path:
        voice=AudioFileClip(voice_path)
        extras.append(voice.volumex(voice_volume))
    if music_path:
        music=AudioFileClip(music_path)
        music=music.audio_loop(duration=30).set_duration(30) if hasattr(music,"audio_loop") else music.set_duration(min(30,music.duration))
        extras.append(music.volumex(music_volume))
    if extras: audio_layers.extend(extras)
    if audio_layers: vertical=vertical.set_audio(CompositeAudioClip(audio_layers).set_duration(30))
    final=vertical
    if captions:
        final=caption_clip_for_segment(model, final, start, 30)
    source_clip.close(); raw.close()
    return final, None

def safe_close(clip):
    try: clip.close()
    except: pass

st.warning("Use only videos you own or are licensed to reuse. Automated exposure screening is imperfect and is not a substitute for moderation or rights checks.")
url=st.text_input("YouTube video URL")
count=st.slider("Number of clips",1,5,3)
original_volume=st.slider("Source audio volume",0.0,2.0,1.0,0.1)
voice_volume=st.slider("Voiceover volume",0.0,2.0,1.0,0.1)
music_volume=st.slider("Background music volume",0.0,1.0,0.25,0.05)
voice_file=st.file_uploader("Optional voiceover/commentary (MP3 or WAV)",type=["mp3","wav"])
music_file=st.file_uploader("Optional background audio (MP3 or WAV)",type=["mp3","wav"])
captions=st.checkbox("Generate and burn auto captions (Whisper tiny)",value=True)

if st.button("Generate Shorts",type="primary",disabled=not url):
    work=tempfile.mkdtemp(prefix="shortmaker_")
    clips=[]
    try:
        with st.spinner("Downloading video (up to 720p)…"):
            source,info=download_video(url,work)
        st.success(f"Downloaded: {info.get('title','Video')}")
        starts=energy_segments(source,count)
        detector=load_nudity_detector()
        model=load_whisper() if captions else None
        voice_path=None; music_path=None
        if voice_file:
            voice_path=os.path.join(work,"voice."+voice_file.name.split(".")[-1]); open(voice_path,"wb").write(voice_file.getvalue())
        if music_file:
            music_path=os.path.join(work,"music."+music_file.name.split(".")[-1]); open(music_path,"wb").write(music_file.getvalue())
        for idx,start in enumerate(starts,1):
            with st.spinner(f"Rendering clip {idx}/{len(starts)}…"):
                video,err=build_short(source,start,voice_path,music_path,original_volume,voice_volume,music_volume,captions,model,detector)
                if err:
                    st.warning(f"Clip {idx}: {err}"); continue
                output=os.path.join(work,f"short_{idx}.mp4")
                video.write_videofile(output,codec="libx264",audio_codec="aac",fps=30,threads=2,preset="ultrafast",logger=None)
                clips.append((output,video))
                st.video(output)
                with open(output,"rb") as f:
                    st.download_button(f"Download clip {idx}",f.read(),file_name=f"short_{idx}.mp4",mime="video/mp4",key=f"dl_{idx}")
                safe_close(video); gc.collect()
        if not clips: st.error("No clips passed the screening/rendering process.")
    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.info("Try a shorter video or fewer clips. Cloud resources may limit long videos.")
    finally:
        for _,c in clips: safe_close(c)
        shutil.rmtree(work,ignore_errors=True)
