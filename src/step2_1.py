import cv2
import numpy as np

# [Step 0] 파일 경로 설정 및 모니터 해상도 대응 리사이즈

path_golden = r"C:\Users\CENOTech\PycharmProjects\PythonProject13\data\pcb_1.jpg"
path_target = r"C:\Users\CENOTech\PycharmProjects\PythonProject13\data\pcb_2.jpg"

img_golden = cv2.imread(path_golden)
img_target = cv2.imread(path_target)

if img_golden is None or img_target is None:
    print(" 이미지를 불러올 수 없습니다.")
    exit()


def resize_to_fit(img, max_width=800):
    h, w = img.shape[:2]
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


img_golden = resize_to_fit(img_golden)
img_target = resize_to_fit(img_target)

gray_golden = cv2.cvtColor(img_golden, cv2.COLOR_BGR2GRAY)
gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

# [Step 1] SIFT 정합

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_golden, None)
kp2, des2 = sift.detectAndCompute(gray_target, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

if len(good_matches) > 10:
    pts_golden = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_target = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_target, pts_golden, cv2.RANSAC, 5.0)

    height, width = img_golden.shape[:2]
    aligned_target = cv2.warpPerspective(img_target, H, (width, height))
    aligned_gray = cv2.cvtColor(aligned_target, cv2.COLOR_BGR2GRAY)

    # 1. SIFT 정합 확인 창
    combined_sift = np.hstack((img_golden, aligned_target))
    cv2.imshow("1. SIFT Alignment Check (Press ANY KEY to continue)", combined_sift)
    print(" [1단계] SIFT 정합 결과 창이 떴습니다. 확인 후 아무 키나 누르세요.")
    cv2.waitKey(0)
    cv2.destroyWindow("1. SIFT Alignment Check (Press ANY KEY to continue)")


    # [Step 2] 마우스 ROI (관심 영역) 지정

    print("\n [2단계] 기판 안쪽(초록색)만 마우스로 드래그하고 ENTER를 누르세요.")
    roi = cv2.selectROI("2. Select ROI (Drag & Press Enter)", img_golden, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("2. Select ROI (Drag & Press Enter)")

    roi_mask = np.zeros_like(gray_golden)
    rx, ry, rw, rh = roi
    if rw > 0 and rh > 0:
        roi_mask[ry:ry + rh, rx:rx + rw] = 255
    else:
        roi_mask[:] = 255


    # [Step 3 & 4] 영상 차분 및 노이즈 제거 (파라미터 튜닝 적용)

    blur_golden = cv2.GaussianBlur(gray_golden, (11, 11), 0)
    blur_target = cv2.GaussianBlur(aligned_gray, (11, 11), 0)

    diff_img = cv2.absdiff(blur_golden, blur_target)
    diff_img = cv2.bitwise_and(diff_img, diff_img, mask=roi_mask)

    # 임계값 30
    _, binary_diff = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)

    # 모폴로지 3x3 커널
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_diff, connectivity=8)

    # [Step 5] 결과 화면 출력 및 창 띄우기 유지

    result_img = aligned_target.copy()
    defect_count = 0

    if rw > 0 and rh > 0:
        cv2.rectangle(result_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # 덩어리 면적 50픽셀 이상만 잡기
        if area > 50:
            defect_count += 1
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result_img, f'Defect {defect_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                        2)

    print(f"\n [최종 결과] 총 {defect_count}개의 차이점이 발견되었습니다.")


    clean_diff_colored = cv2.cvtColor(clean_diff, cv2.COLOR_GRAY2BGR)
    final_view = np.hstack((clean_diff_colored, result_img))

    cv2.imshow("3. Final Result (Press ESC to Exit)", final_view)
    print("3. Final Result (Press ESC to Exit)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print(" SIFT 매칭 실패")