import cv2
import numpy as np
import matplotlib.pyplot as plt


def median_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i] / median_w, 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        else:
            new_peek_ranges.append(peek_range)
    return new_peek_ranges


def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges


def get_font_face_peek_ranges(path_test_image):
    image_color = cv2.imread(path_test_image)
    new_shape = (image_color.shape[1] * 2, image_color.shape[0] * 2)
    image_color = cv2.resize(image_color, new_shape)
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    adaptive_threshold = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    horizontal_sum = np.sum(adaptive_threshold, axis=1)

    plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
    plt.gca().invert_yaxis()
    # plt.show()

    peek_ranges = extract_peek_ranges_from_array(horizontal_sum)

    vertical_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adaptive_threshold[start_y:end_y, :]
        vertical_sum = np.sum(line_img, axis=0)
        vertical_peek_ranges = extract_peek_ranges_from_array(
            vertical_sum,
            minimun_val=40,
            minimun_range=1)
        vertical_peek_ranges2d.append(vertical_peek_ranges)

    vertical_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adaptive_threshold[start_y:end_y, :]
        vertical_sum = np.sum(line_img, axis=0)
        vertical_peek_ranges = extract_peek_ranges_from_array(
            vertical_sum,
            minimun_val=40,
            minimun_range=1)
        vertical_peek_ranges = median_split_ranges(vertical_peek_ranges)
        vertical_peek_ranges2d.append(vertical_peek_ranges)
    return peek_ranges, vertical_peek_ranges2d, image_color



