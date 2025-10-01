import os
import cv2
import pickle
import glob
import numpy as np
from tqdm import tqdm
from stt_engine import create_subtitles, stabilize_speakers, transcribe_audio

class Args:
    def __init__(self):
        self.videoName = 'sample'
        self.videoFolder = 'demo'
        self.savePath = os.path.join(self.videoFolder, self.videoName)
        self.pyframesPath = os.path.join(self.savePath, 'pyframes')
        self.pyworkPath = os.path.join(self.savePath, 'pywork')
        self.pyaviPath = os.path.join(self.savePath, 'pyavi')
        self.nDataLoaderThread = 4

args = Args()

def is_overlap(bbox1, bbox2):
    """
    두 바운딩 박스가 겹치는지 확인합니다.
    bbox = [x_center, y_center, side]
    """
    x1, y1, s1 = bbox1
    x2, y2, s2 = bbox2

    # 박스 경계선 계산
    left1, right1 = x1 - s1, x1 + s1
    top1, bottom1 = y1 - s1, y1 + s1
    left2, right2 = x2 - s2, x2 + s2
    top2, bottom2 = y2 - s2, y2 + s2

    # 겹치지 않는 경우
    if right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1:
        return False
    return True

def create_faces(tracks, scores, args):
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
    
    return faces

def render_subtitles(faces, args, subtitles_data):
    import glob, os, cv2
    from tqdm import tqdm

    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    total_frames = len(flist)

    # 프레임 위치 조회용 dict: frame index -> segment
    frame_to_segment = {}
    for seg in subtitles_data:
        for fidx in range(seg['start_frame'], seg['end_frame'] + 1):
            frame_to_segment[fidx] = seg

    firstImage = cv2.imread(flist[0])
    fh, fw = firstImage.shape[:2]
    vOut = cv2.VideoWriter(
        os.path.join(args.pyaviPath, 'video_rendered.avi'),
        cv2.VideoWriter_fourcc(*'XVID'),
        25,
        (fw, fh)
    )

    last_direction = 'bottom'
    direction_hold_duration = 0
    STABILITY_THRESHOLD = 12
    padding = 20
    bubble_margin = 30

    for fidx, fname in tqdm(enumerate(flist), total=total_frames):
        image = cv2.imread(fname)
        current_faces = faces[fidx]

        seg = frame_to_segment.get(fidx)
        if seg:
            subtitle_text = seg['text']
            speaker_track = seg['speaker']

            # speaker 얼굴 찾기
            speaker_bbox = None
            for face in current_faces:
                if face['track'] == speaker_track:
                    speaker_bbox = (face['x'], face['y'], face['s'])
                    break

            if speaker_bbox:
                x_center, y_center, s = speaker_bbox

                # 다른 얼굴 바운딩 박스
                other_bboxes = [
                    (face['x'], face['y'], face['s'])
                    for face in current_faces if face['track'] != speaker_track
                ]

                # 텍스트 크기
                (text_w, text_h), _ = cv2.getTextSize(subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                bubble_width = text_w + padding * 2
                bubble_height = text_h + padding * 2

                # 여백 계산
                top_dist = y_center - s
                top_space = top_dist - bubble_height
                for obbox in other_bboxes:
                    if is_overlap([x_center, y_center - s - bubble_height/2, bubble_width/2], obbox):
                        top_space = -1
                        break

                bottom_dist = fh - (y_center + s)
                bottom_space = bottom_dist - bubble_height
                for obbox in other_bboxes:
                    if is_overlap([x_center, y_center + s + bubble_height/2, bubble_width/2], obbox):
                        bottom_space = -1
                        break

                left_dist = x_center - s
                left_space = left_dist - bubble_width
                for obbox in other_bboxes:
                    if is_overlap([x_center - s - bubble_width/2, y_center, bubble_width/2], obbox):
                        left_space = -1
                        break

                right_dist = fw - (x_center + s)
                right_space = right_dist - bubble_width
                for obbox in other_bboxes:
                    if is_overlap([x_center + s + bubble_width/2, y_center, bubble_width/2], obbox):
                        right_space = -1
                        break

                spaces = {'top': top_space, 'bottom': bottom_space, 'left': left_space, 'right': right_space}
                valid_spaces = {k: v for k, v in spaces.items() if v >= 0}

                if not valid_spaces:
                    current_direction = 'bottom'
                else:
                    current_direction = max(valid_spaces, key=valid_spaces.get)

                # 안정화
                if current_direction == last_direction:
                    direction_hold_duration += 1
                else:
                    direction_hold_duration = 0

                best_direction_to_render = last_direction
                if direction_hold_duration >= STABILITY_THRESHOLD:
                    best_direction_to_render = current_direction
                    last_direction = current_direction
                    direction_hold_duration = 0

                best_direction = best_direction_to_render

                # --- 말풍선 및 연결선 위치 계산 ---
                line_start_x, line_start_y = int(x_center), int(y_center)
                bubble_top_left, bubble_bottom_right = (0, 0), (0, 0)
                text_pos = (0, 0)
                bubble_margin = 30

                if best_direction == 'top':
                    # 연결선 시작점을 바운딩 박스 상단 중앙으로 설정
                    line_start_y = int(y_center - s)

                    bubble_x = x_center
                    bubble_y = y_center - s - bubble_height // 2 - bubble_margin
                    line_end = (int(bubble_x), int(bubble_y + bubble_height // 2))

                elif best_direction == 'bottom':
                    # 연결선 시작점을 바운딩 박스 하단 중앙으로 설정
                    line_start_y = int(y_center + s)

                    # 기본 bubble 위치
                    bubble_x = x_center
                    bubble_y = y_center + s + bubble_height // 2 + bubble_margin

                    # 화면 안으로 들어오도록 조정
                    bubble_x = max(bubble_width // 2, min(fw - bubble_width // 2, bubble_x))
                    bubble_y = max(bubble_height // 2, min(fh - bubble_height // 2, bubble_y))

                    # 연결선 끝점 (말풍선과 바운딩 박스 연결)
                    line_end = (int(bubble_x), int(bubble_y - bubble_height // 2))

                elif best_direction == 'left':
                    # 연결선 시작점을 바운딩 박스 좌측 중앙으로 설정
                    line_start_x = int(x_center - s)

                    bubble_x = x_center - s - bubble_width // 2 - bubble_margin
                    bubble_y = y_center
                    line_end = (int(bubble_x + bubble_width // 2), int(bubble_y))

                elif best_direction == 'right':
                    # 연결선 시작점을 바운딩 박스 우측 중앙으로 설정
                    line_start_x = int(x_center + s)

                    bubble_x = x_center + s + bubble_width // 2 + bubble_margin
                    bubble_y = y_center
                    line_end = (int(bubble_x - bubble_width // 2), int(bubble_y))
                
                # 말풍선 좌표 계산
                bubble_top_left = (int(bubble_x - bubble_width / 2), int(bubble_y - bubble_height / 2))
                bubble_bottom_right = (int(bubble_x + bubble_width / 2), int(bubble_y + bubble_height / 2))
                text_pos = (bubble_top_left[0] + padding, bubble_top_left[1] + padding + text_h)

               # 6. 렌더링
                # 연결선 그리기
                cv2.line(image, (line_start_x, line_start_y), line_end, (255, 255, 255), 2)

                # 반투명 사각형 그리기
                overlay = image.copy()
                cv2.rectangle(overlay, bubble_top_left, bubble_bottom_right, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

                # 텍스트 그리기
                cv2.putText(image, subtitle_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 항상 프레임 추가
        vOut.write(image)

    vOut.release()

    # 오디오 합치기
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
               (os.path.join(args.pyaviPath, 'video_rendered.avi'),
                os.path.join(args.pyaviPath, 'audio.wav'),
                args.nDataLoaderThread,
                os.path.join(args.pyaviPath, 'final_output.avi')))
    os.system(command)

def main():
    tracks_path = os.path.join(args.pyworkPath, 'tracks.pckl')
    scores_path = os.path.join(args.pyworkPath, 'scores.pckl')

    if not os.path.exists(tracks_path) or not os.path.exists(scores_path):
        print(f"Error: {tracks_path} or {scores_path} not found. Please run the 1st step first.")
        return

    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)

    with open(scores_path, 'rb') as f:
        scores = pickle.load(f)

    faces = create_faces(tracks, scores, args)
    print("Successfully loaded tracks and scores. Starting rendering...")

    # --- 이 부분이 수정됩니다. ---
    print("STT 자막 데이터 생성 시작...")
    # stt_engine.py의 함수를 호출하여 자막 데이터를 생성합니다.
    subtitles_data = create_subtitles(args, faces)
    print("STT 자막 데이터 생성 완료.")
    # ---------------------------

    render_subtitles(faces, args, subtitles_data)
    print("Rendering complete. Final video saved as 'final_output.avi'")

if __name__ == '__main__':
    main()