import os
import whisper
import glob
import re

class Args:
    def __init__(self, pyavi_path, pyframes_path):
        self.pyaviPath = pyavi_path
        self.pyframesPath = pyframes_path

def get_file_paths(args):
    audio_path = os.path.join(args.pyaviPath, 'audio.wav')
    frame_files = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    return audio_path, len(frame_files)

def create_subtitles(args, num_tracks):
    audio_path, num_frames = get_file_paths(args)
    model = whisper.load_model("base")

    result = model.transcribe(audio_path, verbose=False, word_timestamps=True)

    fps = 25
    subtitles_data = {}

    # 긴 segment의 기준 시간 (초)
    LONG_SEGMENT_THRESHOLD = 5.0

    for segment in result['segments']:
        start_time_sec = segment['start']
        end_time_sec = segment['end']

        # segment의 길이가 기준보다 길 경우에만 문장 부호로 분할
        if (end_time_sec - start_time_sec) > LONG_SEGMENT_THRESHOLD:
            words = segment['words']
            if not words:
                continue

            # 문장 부호를 기준으로 단어 리스트를 나눕니다.
            sub_segments = []
            current_sub_segment_words = []

            for word in words:
                current_sub_segment_words.append(word)
                # 텍스트에 문장 부호가 포함되어 있는지 확인
                if re.search(r'[.?!,]', word['text']):
                    sub_segments.append(current_sub_segment_words)
                    current_sub_segment_words = []

            # 마지막 남은 단어들을 추가 (문장 부호가 없었을 경우)
            if current_sub_segment_words:
                sub_segments.append(current_sub_segment_words)

            # 분할된 sub_segments를 자막 데이터로 변환
            for sub_segment_words in sub_segments:
                if not sub_segment_words:
                    continue

                start_frame = int(sub_segment_words[0]['start'] * fps)
                end_frame = int(sub_segment_words[-1]['end'] * fps)
                text = ' '.join([word['text'] for word in sub_segment_words]).strip()

                # 자막 데이터 할당
                for f in range(start_frame, end_frame):
                    if f < num_frames:
                        if f not in subtitles_data:
                            subtitles_data[f] = {}
                        for tidx in range(num_tracks):
                            subtitles_data[f][tidx] = text
        else:
            # segment 길이가 짧으면 기존 방식대로 처리
            start_frame = int(start_time_sec * fps)
            end_frame = int(end_time_sec * fps)
            text = segment['text'].strip()

            for f in range(start_frame, end_frame):
                if f < num_frames:
                    if f not in subtitles_data:
                        subtitles_data[f] = {}
                    for tidx in range(num_tracks):
                        subtitles_data[f][tidx] = text

    return subtitles_data

# if __name__ == "__main__":
#     # --- [여기에 args 객체를 직접 만듭니다] ---
#     # TODO: 1단계에서 생성된 파일 경로에 맞게 수정하세요.
#     test_video_folder = "./demo/sample"
#     test_pyavi_path = os.path.join(test_video_folder, 'pyavi')
#     test_pyframes_path = os.path.join(test_video_folder, 'pyframes')

#     # 임시 args 객체 생성
#     args = Args(test_pyavi_path, test_pyframes_path)

#     # TODO: 테스트할 트랙 개수에 맞게 수정하세요.
#     num_tracks_for_test = 28

#     print("STT 자막 데이터 생성 시작...")

#     # create_subtitles 함수를 호출합니다.
#     subtitles = create_subtitles(args, num_tracks_for_test)

#     print("STT 자막 데이터 생성 완료. 결과 출력:")

#     # 1. 전체 데이터의 쉐입(구조) 출력
#     num_frames_with_subtitles = len(subtitles)
#     # 한 프레임에 여러 트랙의 자막이 있을 수 있으므로, 첫 번째 프레임의 트랙 수를 쉐입으로 간주합니다.
#     first_frame_key = list(subtitles.keys())[0] if subtitles else None
#     num_tracks_in_first_frame = len(subtitles[first_frame_key]) if first_frame_key else 0

#     print(f"\n--- 자막 데이터 쉐입 ---")
#     print(f"전체 프레임 수: {num_frames_with_subtitles}")
#     print(f"트랙 수 (첫 프레임 기준): {num_tracks_in_first_frame}")
#     print("-" * 20)

#     # 2. 0번 트랙의 전체 프레임에 대한 자막 데이터 출력
#     print("\n--- 0번 트랙의 자막 데이터 ---")

#     # subtitles 딕셔너리의 키(프레임 번호)를 정렬합니다.
#     sorted_frames = sorted(subtitles.keys())

#     for frame_num in sorted_frames:
#         # 해당 프레임에 0번 트랙의 자막이 있는 경우에만 출력
#         if 0 in subtitles[frame_num]:
#             print(f"프레임 {frame_num}: {subtitles[frame_num][0]}")

#     print("\n--- 출력 완료. ---")