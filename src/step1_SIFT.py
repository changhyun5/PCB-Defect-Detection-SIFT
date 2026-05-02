import cv2
import numpy as np
import matplotlib.pyplot as plt

path_golden = r"C:\Users\CENOTech\PycharmProjects\PythonProject13\data\pcb_1.jpg"  # 기준 (정상)
path_target = r"C:\Users\CENOTech\PycharmProjects\PythonProject13\data\pcb_2.jpg"  # 검사 대상 (틀어진 각도)

img_golden = cv2.imread(path_golden)
img_target = cv2.imread(path_target)

# BGR을 RGB로 변환
img_golden_rgb = cv2.cvtColor(img_golden, cv2.COLOR_BGR2RGB)
img_target_rgb = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)

# 연산을 위한 흑백 변환
gray_golden = cv2.cvtColor(img_golden, cv2.COLOR_BGR2GRAY)
gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성 및 특징점(Keypoints), 기술자(Descriptors)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_golden, None)
kp2, des2 = sift.detectAndCompute(gray_target, None)

print(f"추출된 특징점 개수 - PCB 1: {len(kp1)}개, PCB 2: {len(kp2)}개")

# 3. 특징점 매칭 (Brute-Force & KNN 알고리즘)
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test: 확실하게 짝지어진 진짜 매칭점(Good Matches)만 필터링
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"필터링 된 유효 매칭점(Good Matches) 개수: {len(good_matches)}개")

# 4. 기하학적 변환 (Homography 및 Warping)
if len(good_matches) > 10:  # 매칭점이 충분할 때만 실행
    # 매칭된 점들의 좌표 분리
    pts_golden = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_target = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC을 이용해 오차를 걸러내고 타겟을 기준으로 맞추는 변환 행렬(H) 계산
    H, mask = cv2.findHomography(pts_target, pts_golden, cv2.RANSAC, 5.0)

    # pcb_2 이미지를 pcb_1의 시점과 크기에 맞게 투영 변환 (Warping)
    height, width = img_golden.shape[:2]
    aligned_target = cv2.warpPerspective(img_target_rgb, H, (width, height))

    # 5. 결과 시각화
    # 정합이 얼마나 잘 되었는지 확인하기 위해 50:50으로 합성 (Alpha Blending)
    blended = cv2.addWeighted(img_golden_rgb, 0.5, aligned_target, 0.5, 0)

    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.title('1. Golden PCB (Reference)')
    plt.imshow(img_golden_rgb)
    plt.axis('off')

    plt.subplot(132)
    plt.title('2. Target PCB (Original Angle)')
    plt.imshow(img_target_rgb)
    plt.axis('off')

    plt.subplot(133)
    plt.title('3. Aligned & Blended Result')
    plt.imshow(blended)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 2단계를 위해 정렬된 이미지를 저장하거나 다음 연산으로 넘깁니다.
    print("성공적으로 좌표 정합이 완료되었습니다. 결과 창에서 이미지가 잘 포개어졌는지 확인하세요.")

else:
    print("매칭점이 부족하여 정합을 수행할 수 없습니다.")