# stt_engine.py (최종 정리 버전)
import os
import whisper
import re
import pickle
import glob
import numpy as np

FPS = 25

# -----------------------
# 1️⃣ Audio → Segment
# -----------------------
def transcribe_audio(audio_path, num_frames, long_segment_threshold=5.0):
    """
    Whisper로 오디오를 텍스트로 변환하고,
    긴 segment는 문장부호 기준으로 sub-segment로 분할
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, verbose=False, word_timestamps=True)

    segments = []

    for seg in result['segments']:
        start_sec = seg['start']
        end_sec = seg['end']
        text = seg['text'].strip()

        # 긴 segment: 문장부호 기준 sub-segment 분할
        if (end_sec - start_sec) > long_segment_threshold and 'words' in seg:
            words = seg['words']
            current_sub = []
            for w in words:
                current_sub.append(w)
                if re.search(r'[.?!,]', w['text']):
                    s_frame = int(current_sub[0]['start'] * FPS)
                    e_frame = int(current_sub[-1]['end'] * FPS)
                    e_frame = min(e_frame, num_frames - 1)
                    segments.append({
                        'start_frame': s_frame,
                        'end_frame': e_frame,
                        'text': ' '.join([x['text'] for x in current_sub]).strip()
                    })
                    current_sub = []
            if current_sub:  # 마지막 남은 단어
                s_frame = int(current_sub[0]['start'] * FPS)
                e_frame = int(current_sub[-1]['end'] * FPS)
                e_frame = min(e_frame, num_frames - 1)
                segments.append({
                    'start_frame': s_frame,
                    'end_frame': e_frame,
                    'text': ' '.join([x['text'] for x in current_sub]).strip()
                })
        else:
            # 짧은 segment는 그대로
            s_frame = int(start_sec * FPS)
            e_frame = int(end_sec * FPS)
            e_frame = min(e_frame, num_frames - 1)
            segments.append({
                'start_frame': s_frame,
                'end_frame': e_frame,
                'text': text
            })
    return segments

# -----------------------
# 2️⃣ Segment Speaker Stabilization
# -----------------------
def stabilize_speakers(segments, faces, threshold=0.5):
    """
    각 segment별 frame 점수(face score) 분석 후 화자를 확정
    faces: frame별 face dict list
    """
    stabilized_segments = []

    for seg in segments:
        start = seg['start_frame']
        end = seg['end_frame']

        # track별 승리 횟수 카운트
        track_win_counts = {}
        for fidx in range(start, end + 1):
            if not faces[fidx]:
                continue
            # 프레임에서 최고 score 가진 face 선택
            best_face = max(faces[fidx], key=lambda f: f['score'])
            if best_face['score'] >= threshold:
                tid = best_face['track']
                track_win_counts[tid] = track_win_counts.get(tid, 0) + 1

        best_tid = max(track_win_counts, key=track_win_counts.get) if track_win_counts else None

        seg_copy = seg.copy()
        seg_copy['speaker'] = best_tid
        stabilized_segments.append(seg_copy)

    return stabilized_segments

# -----------------------
# 3️⃣ Main create_subtitles
# -----------------------
def create_subtitles(args, faces=None):
    """
    args: pyaviPath, pyframesPath
    faces: 실제 트랙 점수 정보 (stabilize_speakers에서 필요)
    
    return: [{'start_frame', 'end_frame', 'text', 'speaker'}, ...]
    """
    audio_path = os.path.join(args.pyaviPath, 'audio.wav')
    num_frames = len(os.listdir(os.path.join(args.pyframesPath)))

    # 1. ASR + Segment 분리
    segments = transcribe_audio(audio_path, num_frames)

    # 2. Segment별 화자 안정화
    if faces is None:
        # faces 정보가 없으면 화자 None 처리
        stabilized_segments = [dict(seg, speaker=None) for seg in segments]
    else:
        stabilized_segments = stabilize_speakers(segments, faces)

    return stabilized_segments

# -----------------------
# Example usage
# -----------------------
class Args:
    def __init__(self, pyavi_path, pyframes_path, pywork_path):
        self.pyaviPath = pyavi_path
        self.pyframesPath = pyframes_path
        self.pyworkPath = pywork_path

def main():
    test_video_folder = "./demo/sample"
    args = Args(os.path.join(test_video_folder, 'pyavi'),
                os.path.join(test_video_folder, 'pyframes'),
                os.path.join(test_video_folder, 'pywork')
                )
    
    tracks_path = os.path.join(args.pyworkPath, 'tracks.pckl')
    scores_path = os.path.join(args.pyworkPath, 'scores.pckl')

    if not os.path.exists(tracks_path) or not os.path.exists(scores_path):
        print(f"Error: {tracks_path} or {scores_path} not found. Please run the 1st step first.")
        return

    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)

    with open(scores_path, 'rb') as f:
        scores = pickle.load(f)

    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))

    faces = [[] for _ in range(len(flist))]
    for tidx, track in enumerate(tracks):
        print(f"Track ID: {tidx}, First frame: {track['track']['frame'].tolist()[0]}")
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s_smoothed = np.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)])
            faces[frame].append({
                'track': tidx,
                'score': float(s_smoothed),
                's': track['proc_track']['s'][fidx],
                'x': track['proc_track']['x'][fidx],
                'y': track['proc_track']['y'][fidx]
            })

    subtitles = create_subtitles(args, faces)

    print("Segments with speaker info:")
    for seg in subtitles:
        print(seg)

if __name__ == "__main__":
    main()
