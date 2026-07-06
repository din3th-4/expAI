import cv2
import os
import shutil


def standardize(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur(image, kernel_size=9):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def low_res(image, scale=0.25):
    h, w = image.shape[:2]

    small = cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_LINEAR
    )

    return cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )


def modify(image, target_size=(256, 256), blur_kernel_size=9, low_res_scale=0.25):
    image = standardize(image, target_size=target_size)

    grayscaled = grayscale(image)
    blurred = blur(image, kernel_size=blur_kernel_size)
    low_res_image = low_res(image, scale=low_res_scale)
    
    return image,grayscaled, blurred, low_res_image


def export(images, output_dir, name_list):
    iteration, emotion, image_type = name_list[0], name_list[1], name_list[2]
    image_types = ["o", "g", "b", "l"]

    if image_type == "o":
        for img, img_name in zip(images, image_types):
            name_list[2] = img_name
            output_filename = f"{('_').join(name_list)}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, img)

            print(f"Saved {output_path}")


def distribute_images(input_dir, output_dir, num_tests=10):
    for i in range(num_tests):
        os.makedirs(os.path.join(output_dir, f"test_{i+1}"), exist_ok=True)

    bases = set()

    for filename in os.listdir(input_dir):
        parts = os.path.splitext(filename)[0].split("_")
        bases.add(f"{parts[0]}_{parts[1]}")

    bases = sorted(bases, key=lambda x: int(x.split("_")[0]))

    for i, base in enumerate(bases):
        start = i % num_tests

        versions = sorted([
            f for f in os.listdir(input_dir)
            if f.startswith(base + "_")
        ])

        for j, filename in enumerate(versions):
            src = os.path.join(input_dir, filename)
            dst = os.path.join(
                output_dir,
                f"test_{(start + j) % num_tests + 1}",
                filename
            )

            shutil.copy2(src, dst)


input_dir = "Original_images"
output_dir = "Modified_images"

os.makedirs(output_dir, exist_ok=True)

extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(extensions):
        input_path = os.path.join(input_dir, filename)

        name_list = os.path.splitext(filename)[0].split("_")
        img = cv2.imread(input_path)

        img_output = modify(img, target_size=(256, 256), blur_kernel_size=9, low_res_scale=0.25)

        export(img_output, output_dir, name_list)

print("Modification Done")

distribute_images(output_dir, "Test_images", num_tests=10)

print("Images Distributed")