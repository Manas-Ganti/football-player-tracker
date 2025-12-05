import cv2
import numpy as np
from typing import Optional, Tuple, List

# Real world circle radius (meters)
CENTER_CIRCLE_RADIUS_M = 9.15

class HomographyEstimator:
    """
    Estimate a per-frame homography from image -> field (meters) using the center circle.
    Usage:
      est = HomographyEstimator(debug=True)
      H_img2field = est.estimate_homography(frame)
      if H is not None:
          field_xy = est.pixel_to_field(H, np.array([[x, y]], dtype=np.float32))  # Nx2 in meters
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def estimate_homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute homography mapping image (px) -> field (meters) for this frame.
        Returns:
          H_img2field: 3x3 homography matrix such that [X_field; Y_field; 1] ~ H_img2field @ [x_img; y_img; 1]
          or None on failure.
        """
        # 1) detect ellipse of center circle
        ellipse = self._detect_center_ellipse(frame)
        if ellipse is None:
            if self.debug:
                print("Ellipse detection failed")
            return None

        (xc, yc), (major_axis, minor_axis), angle = ellipse

        # 2) compute four ellipse boundary points in image coordinates (top,bottom,left,right)
        pts_image = self._ellipse_axis_points((xc, yc), (major_axis, minor_axis), angle)
        # pts_image order: top, bottom, left, right

        # 3) field reference points (meters) in same order
        R = CENTER_CIRCLE_RADIUS_M
        pts_field = np.array([
            [0.0,  R],   # top
            [0.0, -R],   # bottom
            [-R,  0.0],  # left
            [ R,  0.0],  # right
        ], dtype=np.float32)

        pts_image_arr = np.array(pts_image, dtype=np.float32)  # shape (4,2)

        # 4) Solve homography: maps image -> field
        H, mask = cv2.findHomography(pts_image_arr, pts_field, method=cv2.RANSAC, ransacReprojThreshold=10.0)
        if H is None:
            if self.debug:
                print("findHomography failed")
            return None

        # Normalize for numerical stability
        H = H / H[2, 2]

        if self.debug:
            vis = frame.copy()
            for p in pts_image_arr.astype(int):
                cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            # optionally show or save the image
            cv2.imshow("ellipse-homography-debug", vis)
            cv2.waitKey(1)

        return H

    def _detect_center_ellipse(self, frame: np.ndarray) -> Optional[Tuple[Tuple[float,float], Tuple[float,float], float]]:
        """
        Detects the largest white-ish contour likely corresponding to the center circle
        and fits an ellipse. Returns cv2.fitEllipse output: ((xc,yc),(major,minor),angle)
        or None if no suitable ellipse found.
        """
        # convert to HSV to better handle lighting; white has low saturation and high value
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Mask for white: tuned thresholds; adjust if needed
        # white = low saturation, high value
        white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 60, 255]))

        # morphological ops to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return None

        # choose candidate contours by area and approximate circularity (prefer the large near-ellipse)
        best_cnt = None
        best_score = -1.0
        img_area = frame.shape[0] * frame.shape[1]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.0005:   # ignore tiny contours
                continue
            if len(cnt) < 6:
                continue  # fitEllipse needs >= 5 points

            # fitEllipse and estimate how ellipse-like the contour is
            ellipse = cv2.fitEllipse(cnt)
            (xc, yc), (major_axis, minor_axis), angle = ellipse
            # axis ratio near 1 for near-circular; center near image midline? penalize weird shapes
            axis_ratio = max(min(major_axis / (minor_axis + 1e-6), 10.0), 0.1)
            circularity = minor_axis / (major_axis + 1e-6)

            # prefer large and fairly circular contours
            score = area * (1.0 - abs(1.0 - axis_ratio))  # higher if more circular and big

            if score > best_score:
                best_score = score
                best_cnt = cnt

        if best_cnt is None:
            return None

        # Final ellipse fit
        try:
            ellipse = cv2.fitEllipse(best_cnt)
        except Exception as e:
            if self.debug:
                print("fitEllipse exception:", e)
            return None

        return ellipse

    def _ellipse_axis_points(self, center: Tuple[float,float], axes: Tuple[float,float], angle_deg: float) -> List[Tuple[float,float]]:
        """
        Given ellipse ((xc,yc),(major,minor),angle), compute 4 points on ellipse corresponding
        to top, bottom, left, right in the ellipse's local frame (rotated).
        Returns list of (x,y) in image coordinates in order [top, bottom, left, right].
        """
        (xc, yc) = center
        (major_axis, minor_axis) = axes  # OpenCV returns lengths of axes (full), so use /2 for radii
        a = major_axis / 2.0
        b = minor_axis / 2.0
        theta = np.deg2rad(angle_deg)

        # unit ellipse points (local)
        # top in ellipse local coords = (0, -b) if using image coords where y downward, but careful: we'll compute mathematically and then rotate
        # We'll compute points in conventional cartesian and then map to image coordinates (where +y is down)
        # However the ellipse fit returns angle as rotation of the major axis from the x-axis in degrees (OpenCV convention).
        # Compute the four principal points in image coordinates:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Local ellipse points relative to center (x_local, y_local)
        # For robustness use top=(0,-b), bottom=(0,b), left=(-a,0), right=(a,0)
        # Transform: [x_img] = [cos -sin; sin cos] @ [x_local; y_local]
        top = (xc + (0 * cos_t - (-b) * sin_t), yc + (0 * sin_t + (-b) * cos_t))
        bottom = (xc + (0 * cos_t - (b) * sin_t), yc + (0 * sin_t + (b) * cos_t))
        left = (xc + ((-a) * cos_t - 0 * sin_t), yc + ((-a) * sin_t + 0 * cos_t))
        right = (xc + ((a) * cos_t - 0 * sin_t), yc + ((a) * sin_t + 0 * cos_t))

        return [top, bottom, left, right]

    def pixel_to_field(self, H_img2field: np.ndarray, pts_img: np.ndarray) -> np.ndarray:
        """
        Convert image pixel coordinates to field coordinates (meters).
        Args:
          H_img2field: 3x3 homography returned by estimate_homography
          pts_img: Nx2 float32 array of image points [[x,y],...]
        Returns:
          pts_field: Nx2 float32 array [[X_m, Y_m], ...] in meters (field coordinates)
        """
        pts = pts_img.reshape((-1, 1, 2)).astype(np.float32)  # Nx1x2
        pts_field = cv2.perspectiveTransform(pts, H_img2field)  # Nx1x2
        pts_field = pts_field.reshape((-1, 2))
        return pts_field

    def pixel_to_field_single(self, H_img2field: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        arr = np.array([[x, y]], dtype=np.float32)
        res = self.pixel_to_field(H_img2field, arr)
        return float(res[0,0]), float(res[0,1])
