# Python program to read image using OpenCV

# importing OpenCV(cv2) module
import cv2
import numpy as np
import time
import csv


def modify_contrast_and_brightness(img, brightness=0, contrast=100):
    # 上面做法的問題：有做到對比增強，白的的確更白了。
    # 但沒有實現「黑的更黑」的效果
    import math

    # brightness = brightness
    # contrast = contrast # - 減少對比度/+ 增加對比度

    B = brightness / 255.0
    c = contrast / 255.0
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def show_image(img, title="Image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def concat_images(*args):
    concat = np.concatenate([*args], axis=1)
    return concat


def overlap_images(img1, img2, weight1=0.5, weight2=0.5, gamma=0.0):
    dest = cv2.addWeighted(img1, weight1, img2, weight2, gamma)
    return dest


def mask_images(img, mask, invert=False):
    new_mask = mask.copy()
    new_mask = np.sum(new_mask, axis=2)
    if invert:
        inv_0 = new_mask > 0
        inv_1 = new_mask == 0
        new_mask[inv_0] = 0
        new_mask[inv_1] = 255
    else:
        new_mask[new_mask > 0] = 255
    new_mask = new_mask.astype("uint8")
    masked = cv2.bitwise_and(img, img, mask=new_mask)
    return masked


def mask_combine_then_dilate(
    image, mask, kernel_horizontal=(5, 1), kernel_vertical=(1, 5), show_process=False
):
    # mask and contrast
    masked = mask_images(image, mask)
    masked_inv = mask_images(image, mask, invert=True)

    # mask contrast
    contrasted_mask = modify_contrast_and_brightness(masked, brightness=0, contrast=200)
    if show_process:
        show_image(contrasted_mask)

    # invert mask contrast
    contrasted_mask_inv = modify_contrast_and_brightness(
        masked_inv, brightness=0, contrast=250
    )
    if show_process:
        show_image(contrasted_mask_inv)

    # add together
    contrasted_mask_add = contrasted_mask + contrasted_mask_inv
    if show_process:
        show_image(contrasted_mask_add)

    # dilate horizontally
    kernel_horizontal = np.ones(kernel_horizontal, np.uint8)
    contrasted_mask_add_morph_horizontal = cv2.dilate(
        contrasted_mask_add, kernel_horizontal
    )
    contrasted_mask_add_morph_horizontal_f = np.copy(
        contrasted_mask_add_morph_horizontal
    )
    if show_process:
        show_image(contrasted_mask_add_morph_horizontal_f, title="horizontal kernel")

    # dilate vertically
    kernel_vertical = np.ones(kernel_vertical, np.uint8)
    contrasted_mask_add_morph_vertical = cv2.dilate(
        contrasted_mask_add, kernel_vertical
    )
    contrasted_mask_add_morph_vertical_f = np.copy(contrasted_mask_add_morph_vertical)
    if show_process:
        show_image(contrasted_mask_add_morph_vertical, title="vertical kernel")

    # combine all images
    contrasted_final = (
        contrasted_mask_add
        + contrasted_mask_add_morph_horizontal_f
        + contrasted_mask_add_morph_vertical_f
    )
    contrasted_final[contrasted_final > 255] = 255
    if show_process:
        show_image(
            modify_contrast_and_brightness(contrasted_final, 0, 200),
            title="all added together",
        )

    # compare
    if show_process:
        show_image(
            concat_images(contrasted_final, contrasted_mask_add),
            title="compare image with one before dilation",
        )

    # for display
    concat = concat_images(overlap_images(mask, contrasted_final, 0.5, 0.5), image)
    if show_process:
        show_image(concat, "compare with sample")

    contrasted_final = modify_contrast_and_brightness(
        contrasted_final, brightness=0, contrast=250
    )

    contrasted_final = mask_images(contrasted_final, mask)

    return contrasted_final


def mask_dilate_then_combine(
    image, mask, kernel_horizontal=(5, 1), kernel_vertical=(1, 5), show_process=False
):
    # mask and contrast
    masked = mask_images(image, mask)
    masked_inv = mask_images(image, mask, invert=True)
    kernel_horizontal = np.ones(kernel_horizontal, np.uint8)
    kernel_vertical = np.ones(kernel_vertical, np.uint8)

    # mask contrast
    contrasted_mask = modify_contrast_and_brightness(masked, brightness=0, contrast=225)
    contrasted_mask_before = np.copy(contrasted_mask)
    if show_process:
        show_image(contrasted_mask)
    # dilate
    contrasted_mask_horizontal = cv2.dilate(contrasted_mask, kernel_horizontal)
    if show_process:
        show_image(contrasted_mask_horizontal, title="horizontal kernel")
    # dilate vertically
    contrasted_mask_vertical = cv2.dilate(contrasted_mask, kernel_vertical)
    if show_process:
        show_image(contrasted_mask_vertical, title="vertical kernel")

    # combine all images
    contrasted_mask = (
        contrasted_mask + contrasted_mask_horizontal + contrasted_mask_vertical
    )
    contrasted_mask[contrasted_mask > 255] = 255
    contrasted_mask = modify_contrast_and_brightness(
        contrasted_mask, brightness=0, contrast=250
    )
    if show_process:
        show_image(contrasted_mask, title="all added together mask")

    contrasted_mask_morphed = cv2.morphologyEx(
        contrasted_mask, cv2.MORPH_OPEN, kernel=(4, 4), iterations=1
    )
    if show_process:
        show_image(contrasted_mask_morphed, title="together morphed")

    if show_process:
        show_image(
            concat_images(
                contrasted_mask_before, contrasted_mask, contrasted_mask_morphed
            ),
            title="Compared_all",
        )

    # mask contrast inverted (We do not dilate the inverted masked image)
    contrasted_mask_inv = modify_contrast_and_brightness(
        masked_inv, brightness=0, contrast=225
    )
    contrasted_mask_inv_before = np.copy(contrasted_mask_inv)
    if show_process:
        show_image(contrasted_mask_inv)

    # add together
    contrasted_mask_final = contrasted_mask_morphed + contrasted_mask_inv
    if show_process:
        show_image(contrasted_mask_final, "combined mask and inv")

    contrasted_mask_final = modify_contrast_and_brightness(
        contrasted_mask_final, brightness=0, contrast=250
    )
    contrasted_mask_final = mask_images(contrasted_mask_final, mask)

    contrasted_without_dilation = contrasted_mask_before + contrasted_mask_inv_before
    # compare
    if show_process:
        show_image(
            concat_images(contrasted_without_dilation, contrasted_mask_final),
            title="compare image with one before dilation",
        )

    # # for display
    concat = concat_images(overlap_images(mask, contrasted_mask_final, 0.5, 0.5), image)
    if show_process:
        show_image(concat, "compare with sample")

    return contrasted_mask_final


def findDefectTu(detect, min_size=4, max_size=1024):
    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
        detect
    )
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below.
    sizes = sizes[1:]
    nb_blobs -= 1

    # minimum size of particles we want to keep (number of pixels).
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    # max_size = 1024
    # min_size = 4

    # output image with only the kept components
    im_result = np.zeros((detect.shape), dtype=np.uint8)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size and sizes[blob] < max_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255

    return im_result


"""
base: the image will draw on
im_result: the defect graph
shift: add padding rectangle size
"""


def drawRectangle(im_result, shift=10, color_column=2):

    im_result = im_result.astype(np.uint8)
    contours, _ = cv2.findContours(im_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    # drawing = base
    drawing = np.zeros((im_result.shape[0], im_result.shape[1], 3), dtype=np.uint8)
    # drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)
    for i in range(len(contours)):
        if boundRect[i][0] == 0 or boundRect[i][1] == 0:
            continue
        color = [255, 255, 255]
        # cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(
            drawing,
            (int(boundRect[i][0]) - shift, int(boundRect[i][1]) - shift),
            (
                int(boundRect[i][0] + boundRect[i][2]) + shift,
                int(boundRect[i][1] + boundRect[i][3]) + shift,
            ),
            color,
            -1,
        )
        # cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    # do second time to merge rectangle
    shift = 0
    drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(drawing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    # drawing = base
    drawing = np.zeros((drawing.shape[0], drawing.shape[1], 3), dtype=np.uint8)
    # drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)
    for i in range(len(contours)):
        if boundRect[i][0] == 0 or boundRect[i][1] == 0:
            continue
        color = [0, 0, 0]
        color[color_column] = 255
        # cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(
            drawing,
            (int(boundRect[i][0]) - shift, int(boundRect[i][1]) - shift),
            (
                int(boundRect[i][0] + boundRect[i][2]) + shift,
                int(boundRect[i][1] + boundRect[i][3]) + shift,
            ),
            color,
            4,
        )
        # cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    return drawing, len(contours)


def overlap_images(img1, img2, weight1=0.5, weight2=0.5, gamma=0.0):
    dest = cv2.addWeighted(img1, weight1, img2, weight2, gamma)
    return dest


def detect_sag(sem, gds):

    dic = {}
    # build mask
    ret, mask = cv2.threshold(gds, 10, 255, cv2.THRESH_BINARY)

    # sem 上 mask
    sem_mask = cv2.bitwise_and(sem, sem, mask=mask)
    dic["sem_mask"] = cv2.cvtColor(sem_mask, cv2.COLOR_GRAY2BGR)

    # 分界線 與 外部 二值化成白色
    # 內部 二值化成黑色
    th, boundary_external_white = cv2.threshold(sem_mask, 65, 255, cv2.THRESH_BINARY)
    boundary_external_white = cv2.medianBlur(boundary_external_white, 9)
    boundary_external_white = cv2.erode(
        boundary_external_white, np.ones((8, 8), np.uint8)
    )
    dic["boundary_external_white"] = cv2.cvtColor(
        boundary_external_white, cv2.COLOR_GRAY2BGR
    )

    # 分界線 二值化成黑色
    # 內部 與 外部 二值化成白色
    th, boundary_white = cv2.threshold(sem_mask, 100, 255, cv2.THRESH_BINARY)
    boundary_white = cv2.bitwise_not(boundary_white)
    boundary_white = cv2.medianBlur(boundary_white, 7)
    boundary_white = cv2.bitwise_not(boundary_white)
    boundary_white = cv2.dilate(boundary_white, np.ones((1, 1), np.uint8))
    boundary_black = cv2.bitwise_not(boundary_white)
    dic["boundary_black"] = cv2.cvtColor(boundary_black, cv2.COLOR_GRAY2BGR)

    # 將兩張疊合
    # 產生出只有缺陷為白色
    sag = cv2.bitwise_and(
        boundary_external_white, boundary_external_white, mask=boundary_black
    )
    for i in range(3):
        sag = cv2.medianBlur(sag, 5)
    dic["sag"] = cv2.cvtColor(sag, cv2.COLOR_GRAY2BGR)

    # 比較成效
    result = overlap_images(sag, sem_mask)
    dic["result"] = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # 畫框框
    rect, count = drawRectangle(cv2.cvtColor(dic["sag"], cv2.COLOR_BGR2GRAY))
    dic["rect"] = rect
    dic["r_rect_count"] = count
    return dic


def detect_convex(sem, gds):
    """
    extract out area
    """
    dic = {}
    sem_ = sem.copy()
    sem_ = cv2.medianBlur(sem_, 5)
    th, sem_ = cv2.threshold(sem_, 100, 255, cv2.THRESH_BINARY)
    dic["convex_edge"] = cv2.cvtColor(sem_, cv2.COLOR_GRAY2BGR)
    sem_ = cv2.erode(sem_, None, iterations=2)
    sem_ = cv2.dilate(sem_, None, iterations=2)
    gds = cv2.dilate(gds, None, iterations=2)
    out_area = overlap_images(sem_, gds, 1, 1)
    out_area = cv2.bitwise_not(out_area)

    # cv2.imshow("out_area", out_area)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    """
    find defect, turn to small white area
    """
    edges = findDefectTu(out_area)
    # cv2.imshow("Canny", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    dic["convex_point"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    """
    remove big white area and draw rectangle
    """
    drawing, count = drawRectangle(edges, color_column=1)
    dic["convex_rect"] = drawing
    dic["g_rect_count"] = count

    gds = cv2.cvtColor(gds, cv2.COLOR_GRAY2BGR)
    sem = cv2.cvtColor(sem, cv2.COLOR_GRAY2BGR)
    gds = overlap_images(gds, sem, 0.3, 1)
    rec = overlap_images(gds, drawing, 0.3, 1)
    drawing = cv2.add(drawing, rec)
    dic["convex_res"] = drawing

    return dic


# process_image: 1024, 1024, 3
# ground_truth: 1024, 1024, 3
# output: 1024, 1024
def detect_defect(image, ground_truth, output, limit=6):
    edge_len = 40

    image = np.sum(image, axis=2) / 3
    output = np.sum(output, axis=2) / 3
    # ground_truth = np.sum(ground_truth, axis=2) / 3

    defect = np.zeros(image.shape, dtype=np.uint8)
    for i in range(1, image.shape[0] - 1):
        # print(i)
        for j in range(1, image.shape[1] - 1):
            if output[i, j] == 255:
                continue
            if (ground_truth[i : i + limit, j] == 0).all() and (
                ground_truth[i : i + limit, j + 1] == 255
            ).all():
                index = j + 1
                while (
                    index < image.shape[1]
                    and (ground_truth[i : i + limit, index] == 255).all()
                ):
                    if (image[i : i + limit, index] == 255).any():
                        break
                    index += 1
                if index - j < edge_len and index - j > 5:
                    defect[i : i + limit, j + 1 : index] = 255
            if (ground_truth[i : i + limit, j] == 0).all() and (
                ground_truth[i : i + limit, j - 1] == 255
            ).all():
                index = j - 1
                while index >= 0 and (ground_truth[i : i + limit, index] == 255).all():
                    if (image[i : i + limit, index] == 255).any():
                        break
                    index -= 1
                if j - index < edge_len and j - index > 5:
                    defect[i : i + limit, index + 1 : j] = 255
            if (ground_truth[i, j : j + limit] == 0).all() and (
                ground_truth[i + 1, j : j + limit] == 255
            ).all():
                index = i + 1
                while (
                    index < image.shape[0]
                    and (ground_truth[index, j : j + limit] == 255).all()
                ):
                    if (image[index, j : j + limit] == 255).any():
                        break
                    index += 1
                if index - i < edge_len and index - i > 5:
                    defect[i + 1 : index, j : j + limit] = 255
            if (ground_truth[i, j : j + limit] == 0).all() and (
                ground_truth[i - 1, j : j + limit] == 255
            ).all():
                index = i - 1
                while index >= 0 and (ground_truth[index, j : j + limit] == 255).all():
                    if (image[index, j : j + limit] == 255).any():
                        break
                    index -= 1
                if i - index < edge_len and i - index > 5:
                    defect[index + 1 : i, j : j + limit] = 255

    image = np.stack((image,) * 3, axis=-1)
    ground_truth = np.stack((ground_truth,) * 3, axis=-1)
    defect = np.stack((defect,) * 3, axis=-1)
    defect[:, :, 0:2] = 0

    return {"convex_inside_defect": defect}


def rect_mask(sem, rect):
    grey = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(sem, sem, mask=mask_inv)
    img2_fg = cv2.bitwise_and(rect, rect, mask=mask)
    return cv2.add(img1_bg, img2_fg)


def process_images(sem3d, gds3d) -> dict:
    sem = cv2.cvtColor(sem3d, cv2.COLOR_BGR2GRAY)
    gds = cv2.cvtColor(gds3d, cv2.COLOR_BGR2GRAY)
    return_sag = detect_sag(sem, gds)
    return_convex = detect_convex(sem, gds)
    process_dic = {**return_sag, **return_convex}
    return_dic = {}
    return_dic["r_rect_count"] = int(process_dic["r_rect_count"])
    return_dic["g_rect_count"] = int(process_dic["g_rect_count"])
    return_dic["sem_gds"] = overlap_images(sem3d, gds3d, 0.7, 0.3)
    return_dic["sem_gds_r_rect"] = rect_mask(return_dic["sem_gds"], process_dic["rect"])
    return_dic["sem_gds_g_rect"] = rect_mask(
        return_dic["sem_gds"], process_dic["convex_rect"]
    )
    return_dic["sem_gds_rg_rect"] = rect_mask(
        return_dic["sem_gds_r_rect"], process_dic["convex_rect"]
    )
    return_dic["sem_r_rect"] = rect_mask(sem3d, process_dic["rect"])
    return_dic["sem_g_rect"] = rect_mask(sem3d, process_dic["convex_rect"])
    return_dic["sem_rg_rect"] = rect_mask(
        return_dic["sem_r_rect"], process_dic["convex_rect"]
    )

    return return_dic
    for key in return_dic.keys():
        print(key)
        # cv2.imshow(key,return_dic[key])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    IMG_PATH = "resource/"
    OUT_PATH = "out/"
    for i in range(1, 2):
        ID = i
        gds_3d = cv2.imread(IMG_PATH + f"gds_{ID}.png")
        sem_3d = cv2.imread(IMG_PATH + f"sem_{ID}.png")
        process_images(sem_3d, gds_3d)
        continue
        gds = cv2.imread(IMG_PATH + f"gds_{ID}.png", cv2.IMREAD_GRAYSCALE)
        sem = cv2.imread(IMG_PATH + f"sem_{ID}.png", cv2.IMREAD_GRAYSCALE)

        return_sag = detect_sag(sem, gds)
        return_convex = detect_convex(sem, gds)

        image = mask_combine_then_dilate(sem_3d, gds_3d)
        # return_defect = detect_defect(image, gds, return_convex['convex_point'])

        # return_dic = {**return_sag, **return_convex, **return_defect}
        return_dic = {**return_sag, **return_convex}
        types1 = [k.shape for k in return_dic.values()]
        for key in return_dic.keys():
            cv2.imshow(key, return_dic[key])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        gds_sem = overlap_images(gds, sem, 1, 0.7)
        mix = overlap_images(return_dic["convex_rect"], return_dic["rect"], 1, 1)
        show_image(overlap_images(mix, sem_3d, 1, 0.7))
