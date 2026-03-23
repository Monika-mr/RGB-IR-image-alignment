import os
import SimpleITK as sitk
import numpy as np
import cv2
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

def align_and_save(rgb_image, ir_image, save_path, overlay_path=None):
    # Step 1: Metadata sync
    ir_image.SetSpacing(rgb_image.GetSpacing())
    ir_image.SetOrigin(rgb_image.GetOrigin())
    ir_image.SetDirection(rgb_image.GetDirection())

    # Step 2: Convert RGB to grayscale (green channel)
    rgb_gray = sitk.VectorIndexSelectionCast(rgb_image, 1, sitk.sitkFloat32)

    # Step 3: Global Alignment using Similarity2D
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(1.0, 0.01, 300, 1e-6)
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    initial_transform = sitk.Similarity2DTransform()
    initial_transform = sitk.CenteredTransformInitializer(
        rgb_gray, ir_image, initial_transform, sitk.CenteredTransformInitializerFilter.GEOMETRY)
    initial_transform.SetScale(0.95)
    initial_transform.SetAngle(0.0)
    initial_transform.SetTranslation((0.0, 0.0))

    registration.SetInitialTransform(initial_transform, inPlace=False)
    registration.SetOptimizerScalesFromPhysicalShift()
    final_transform = registration.Execute(rgb_gray, ir_image)

    ir_global_aligned = sitk.Resample(ir_image, rgb_gray, final_transform,
                                      sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    # Step 4: Local BSpline Warp (elbow region)
    mask_array = sitk.GetArrayFromImage(rgb_gray)
    mask_array[:] = 0
    mask_array[390:470, 200:350] = 1
    mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask.CopyInformation(rgb_gray)

    bspline_transform = sitk.BSplineTransformInitializer(
        image1=rgb_gray, transformDomainMeshSize=[6, 6], order=3)

    params = np.array(bspline_transform.GetParameters())
    for idx in range(0, len(params), 2):
        x_idx = idx
        y_idx = idx + 1
        if 10 < x_idx < 50 and 20 < y_idx < 60:
            params[x_idx] -= 0.8
    bspline_transform.SetParameters(tuple(params))

    ir_shifted = sitk.Resample(ir_global_aligned, rgb_gray, bspline_transform,
                               sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    # Step 5: TPS Finger + Watch Warp
    rgb_np = sitk.GetArrayFromImage(rgb_gray)
    ir_np = sitk.GetArrayFromImage(ir_shifted)

    src_points = np.array([
        [245, 430], [270, 420], [300, 440],
        [330, 460], [210, 470], [215, 455]
    ])
    dst_points = np.array([
        [250, 425], [275, 415], [305, 435],
        [335, 455], [215, 465], [220, 450]
    ])

    def tps_warp(src_pts, dst_pts, image):
        h, w = image.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x_flat = grid_x.flatten()
        grid_y_flat = grid_y.flatten()
        rbf_x = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0], function='thin_plate')
        rbf_y = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 1], function='thin_plate')
        map_x = rbf_x(grid_x_flat, grid_y_flat).reshape((h, w)).astype(np.float32)
        map_y = rbf_y(grid_x_flat, grid_y_flat).reshape((h, w)).astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    ir_tps_aligned = tps_warp(src_points, dst_points, ir_np)
    ir_final = sitk.GetImageFromArray(ir_tps_aligned.astype(np.float32))
    ir_final.CopyInformation(rgb_gray)
    sitk.WriteImage(ir_final, save_path)

    # Optional overlay image
    if overlay_path:
        blend = cv2.addWeighted(rgb_np, 0.5, ir_tps_aligned, 0.5, 0)
        plt.imsave(overlay_path, blend, cmap='gray')


# === Batch Processing ===
rgb_folder = r"C:\Users\mrmon\Downloads\Four_channel\RGB_images\train\images"
ir_folder = r"C:\Users\mrmon\Downloads\Four_channel\IR_images\train\images"
output_folder = r"C:\Users\mrmon\Downloads\Four_channel\Aligned_Output"
overlay_folder = r"C:\Users\mrmon\Downloads\Four_channel\overlays"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(overlay_folder, exist_ok=True)

rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith(".jpg")]

for rgb_file in rgb_files:
    rgb_path = os.path.join(rgb_folder, rgb_file)
    ir_file = rgb_file.replace("jpg", "jpg").replace(".rf.", ".rf.")
    # Remove .rf.hash from filename and find any matching IR file
    rgb_base = rgb_file.split(".rf.")[0]  # frame_00000_jpg
    matching_ir = [f for f in os.listdir(ir_folder) if rgb_base in f]

    if not matching_ir:
        print(f"Skipping {rgb_file}: matching IR file not found.")
        continue

    ir_file = matching_ir[0]
    ir_path = os.path.join(ir_folder, ir_file)
    
    if not os.path.exists(ir_path):
        print(f"Skipping {rgb_file}: matching IR file not found.")
        continue

    print(f"Processing: {rgb_file}")
    rgb_image = sitk.ReadImage(rgb_path, sitk.sitkVectorUInt8)
    ir_image = sitk.ReadImage(ir_path, sitk.sitkFloat32)

    output_path = os.path.join(output_folder, f"{os.path.splitext(rgb_file)[0]}_aligned.nii")
    overlay_path = os.path.join(overlay_folder, f"{os.path.splitext(rgb_file)[0]}_blend.png")

    align_and_save(rgb_image, ir_image, output_path, overlay_path)
