import os
import cv2
import pickle
import glob
import numpy as np
from tqdm import tqdm
from stt_engine import create_subtitles

# ArgumentParser 대신 간단한 설정 변수를 사용합니다.
class Args:
    def __init__(self):
        # TODO: 1단계에서 사용한 videoName과 videoFolder에 맞게 수정하세요.
        self.videoName = 'sample'
        self.videoFolder = 'demo'
        self.savePath = os.path.join(self.videoFolder, self.videoName)
        self.pyframesPath = os.path.join(self.savePath, 'pyframes')
        self.pyworkPath = os.path.join(self.savePath, 'pywork')
        self.pyaviPath = os.path.join(self.savePath, 'pyavi')
        self.nDataLoaderThread = 4

args = Args()

# --- 헬퍼 함수 ---
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


def render_subtitles(tracks, scores, args, subtitles_data):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()

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

    firstImage = cv2.imread(flist[0])
    fw, fh = firstImage.shape[1], firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_rendered.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh))

    # === 안정화 로직에 필요한 변수들 ===
    last_direction = 'bottom'  # 초기값 설정
    direction_hold_duration = 0  # 현재 방향이 유지된 프레임 수
    STABILITY_THRESHOLD = 12   # 0.5초 (25fps 기준 12프레임)

    for fidx, fname in tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)
        current_faces = faces[fidx]

        if not current_faces:
            vOut.write(image)
            continue

        best_speaker = max(current_faces, key=lambda f: f['score'], default=None)

        if best_speaker and best_speaker['score'] >= 0.5 and fidx in subtitles_data and best_speaker['track'] in subtitles_data[fidx]:
            speaker_bbox = (best_speaker['x'], best_speaker['y'], best_speaker['s'])
            subtitle_text = subtitles_data[fidx][best_speaker['track']]

            # --- 말풍선 위치 계산 (가장 넓은 여백 찾기) ---
            other_bboxes = [
                (face['x'], face['y'], face['s'])
                for face in current_faces if face['track'] != best_speaker['track']
            ]

            (text_w, text_h), _ = cv2.getTextSize(subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            padding = 20
            bubble_width = text_w + padding * 2
            bubble_height = text_h + padding * 2

            x_center, y_center, s = speaker_bbox

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

            # 5. 가장 넓은 여백 선택
            spaces = {'top': top_space, 'bottom': bottom_space, 'left': left_space, 'right': right_space}
            valid_spaces = {k: v for k, v in spaces.items() if v >= 0}

            if not valid_spaces:
                current_direction = 'bottom'
            else:
                current_direction = max(valid_spaces, key=valid_spaces.get)

            # === 안정화 로직 ===
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

                bubble_x = x_center
                bubble_y = y_center + s + bubble_height // 2 + bubble_margin
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

        # 바운딩 박스 그리기 (주석 처리 또는 삭제)
        # for face in current_faces:
        #     clr_box = (0, 255, 0) if face == best_speaker and face['score'] >= 0.5 else (0, 0, 255)
        #     cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])), clr_box, 10)

        vOut.write(image)

    vOut.release()

    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
           (os.path.join(args.pyaviPath, 'video_rendered.avi'), os.path.join(args.pyaviPath, 'audio.wav'),
            args.nDataLoaderThread, os.path.join(args.pyaviPath, 'final_output.avi')))
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

    print("Successfully loaded tracks and scores. Starting rendering...")

    # --- 이 부분이 수정됩니다. ---
    print("STT 자막 데이터 생성 시작...")
    # stt_engine.py의 함수를 호출하여 자막 데이터를 생성합니다.
    subtitles_data = create_subtitles(args, len(tracks))
    print("STT 자막 데이터 생성 완료.")
    # ---------------------------

    render_subtitles(tracks, scores, args, subtitles_data)
    print("Rendering complete. Final video saved as 'final_output.avi'")

if __name__ == '__main__':
    main()