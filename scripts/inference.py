# pylint: disable=E1101
# scripts/inference.py

"""
This script contains the main inference pipeline for processing audio and image inputs to generate a video output.

The script imports necessary packages and classes, defines a neural network model, 
and contains functions for processing audio embeddings and performing inference.

The main inference process is outlined in the following steps:
1. Initialize the configuration.
2. Set up runtime variables.
3. Prepare the input data for inference (source image, face mask, and face embeddings).
4. Process the audio embeddings.
5. Build and freeze the model and scheduler.
6. Run the inference loop and save the result.

Usage:
This script can be run from the command line with the following arguments:
- audio_path: Path to the audio file.
- image_path: Path to the source image.
- face_mask_path: Path to the face mask image.
- face_emb_path: Path to the face embeddings file.
- output_path: Path to save the output video.

Example:
python scripts/inference.py --audio_path audio.wav --image_path image.jpg 
    --face_mask_path face_mask.png --face_emb_path face_emb.pt --output_path output.mp4
"""

import argparse
import gc
import os

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
import torch
from torch.profiler import profile, record_function, ProfilerActivity


from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.util import tensor_to_video


# from diffusers.utils.import_utils import is_xformers_available


class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    
    Args:
        reference_unet (UNet2DConditionModel): The UNet2DConditionModel used as a reference for inference.
        denoising_unet (UNet3DConditionModel): The UNet3DConditionModel used for denoising the input audio.
        face_locator (FaceLocator): The FaceLocator model used to locate the face in the input image.
        imageproj (nn.Module): The ImageProjector model used to project the source image onto the face.
        audioproj (nn.Module): The AudioProjector model used to project the audio embeddings onto the face.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(self,):
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }


def process_audio_emb(audio_emb):
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb


# Function to handle the setup of pixel values with motion frames
def prepare_pixel_values(tensor_result, source_image_pixels, config):
    if len(tensor_result) == 0:
        # The first iteration
        motion_zeros = source_image_pixels.repeat(config.data.n_motion_frames, 1, 1, 1)
        motion_zeros = motion_zeros.to(dtype=source_image_pixels.dtype, device=source_image_pixels.device)
        pixel_values_ref_img = torch.cat([source_image_pixels, motion_zeros], dim=0)
    else:
        motion_frames = tensor_result[-1][0].permute(1, 0, 2, 3)  # Combining permute and slicing
        motion_frames = motion_frames[-config.data.n_motion_frames:]  # Only select last n_motion_frames
        motion_frames = (motion_frames * 2.0 - 1.0).to(dtype=source_image_pixels.dtype, device=source_image_pixels.device)
        pixel_values_ref_img = torch.cat([source_image_pixels, motion_frames], dim=0)

    return pixel_values_ref_img.unsqueeze(0)


# Main inference function
def inference_step(t, clip_length, audio_emb, net, config, source_image_pixels, source_image_face_emb, 
                   source_image_face_region, source_image_full_mask, source_image_face_mask, 
                   source_image_lip_mask, img_size, pipeline, tensor_result, generator, motion_scale, times):

    # Prepare the pixel values reference image
    pixel_values_ref_img = prepare_pixel_values(tensor_result, source_image_pixels, config)

    # Prepare the audio tensor
    audio_tensor = audio_emb[t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])]
    audio_tensor = audio_tensor.unsqueeze(0).to(device=net.audioproj.device, dtype=net.audioproj.dtype)
    audio_tensor = net.audioproj(audio_tensor)

    # Run the pipeline
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    pipeline_output = pipeline(
        ref_image=pixel_values_ref_img,
        audio_tensor=audio_tensor,
        face_emb=source_image_face_emb,
        face_mask=source_image_face_region,
        pixel_values_full_mask=source_image_full_mask,
        pixel_values_face_mask=source_image_face_mask,
        pixel_values_lip_mask=source_image_lip_mask,
        width=img_size[0],
        height=img_size[1],
        video_length=clip_length,
        num_inference_steps=config.inference_steps,
        guidance_scale=config.cfg_scale,
        generator=generator,
        motion_scale=motion_scale,
    )
    # print("CPU time")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    # input("Press Enter to continue...")

    tensor_result.append(pipeline_output.videos)

    print(f"inference {t + 1} / {times}")


def inference_process(args: argparse.Namespace, setting_steps=40, setting_cfg=3.5, settings_seed=42, settings_fps=25, settings_motion_pose_scale=1.1, settings_motion_face_scale=1.1, settings_motion_lip_scale=1.1, settings_n_motion_frames=2, settings_n_sample_frames=16):
    """
    Perform inference processing.

    Args:
        args (argparse.Namespace): Command-line arguments.

    This function initializes the configuration for the inference process. It sets up the necessary
    modules and variables to prepare for the upcoming inference steps.
    """
    # 1. init config
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, vars(args))
    
    
    if setting_steps is not None:
        config.inference_steps = setting_steps
    if setting_cfg is not None:
        config.cfg_scale = setting_cfg
    if settings_seed is not None:
        config.seed = int(settings_seed)
    if settings_fps is not None:
        config.data.export_video.fps = settings_fps
    if settings_motion_pose_scale is not None:
        config.pose_weight = settings_motion_pose_scale
    if settings_motion_face_scale is not None:
        config.face_weight = settings_motion_face_scale
    if settings_motion_lip_scale is not None:
        config.lip_weight = settings_motion_lip_scale
    if settings_n_motion_frames is not None:
        config.data.n_motion_frames = settings_n_motion_frames
    if settings_n_sample_frames is not None:
        config.data.n_sample_frames = settings_n_sample_frames
    
    
    source_image_path = config.source_image
    driving_audio_path = config.driving_audio
    save_path = config.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    motion_scale = [config.pose_weight, config.face_weight, config.lip_weight]
    if args.checkpoint is not None:
        config.audio_ckpt_dir = args.checkpoint
    # 2. runtime variables
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32

    # 3. prepare inference data
    # 3.1 prepare source image, face mask, face embeddings
    img_size = (config.data.source_image.width,
                config.data.source_image.height)
    clip_length = config.data.n_sample_frames
    face_analysis_model_path = config.face_analysis.model_path
    with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
        source_image_pixels, \
        source_image_face_region, \
        source_image_face_emb, \
        source_image_full_mask, \
        source_image_face_mask, \
        source_image_lip_mask = image_processor.preprocess(
            source_image_path, save_path, config.face_expand_ratio)

    # 3.2 prepare audio embeddings
    sample_rate = config.data.driving_audio.sample_rate
    assert sample_rate == 16000, "audio sample rate must be 16000"
    fps = config.data.export_video.fps
    wav2vec_model_path = config.wav2vec.model_path
    wav2vec_only_last_features = config.wav2vec.features == "last"
    audio_separator_model_file = config.audio_separator.model_path
    with AudioProcessor(
        sample_rate,
        fps,
        wav2vec_model_path,
        wav2vec_only_last_features,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        os.path.join(save_path, "audio_preprocess")
    ) as audio_processor:
        audio_emb = audio_processor.preprocess(driving_audio_path)
        
    # Clear memory 
    # del image_processor
    del audio_processor
    gc.collect()
    torch.cuda.empty_cache()
    

    # 4. build modules
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    vae = AutoencoderKL.from_pretrained(config.vae.model_path)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.base_model_path, subfolder="unet")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs),
        use_landmark=False,
    )
    face_locator = FaceLocator(conditioning_embedding_channels=320)
    image_proj = ImageProjModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    )

    audio_proj = AudioProjModel(
        seq_len=5,
        blocks=12,  # use 12 layers' hidden states of wav2vec
        channels=768,  # audio embedding channel
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device=device, dtype=weight_dtype)

    audio_ckpt_dir = config.audio_ckpt_dir


    # Freeze
    vae.requires_grad_(False)
    image_proj.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    face_locator.requires_grad_(False)
    audio_proj.requires_grad_(False)
    
    # Not working soryy :(
    # if is_xformers_available():
    #     reference_unet.enable_xformers_memory_efficient_attention()
    #     denoising_unet.enable_xformers_memory_efficient_attention()

    # reference_unet.enable_gradient_checkpointing()
    # denoising_unet.enable_gradient_checkpointing()
    
    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        image_proj,
        audio_proj,
    )

    m,u = net.load_state_dict(
        torch.load(
            os.path.join(audio_ckpt_dir, "net.pth"),
            map_location="cpu",
        ),
    )
    assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
    print("loaded weight from ", os.path.join(audio_ckpt_dir, "net.pth"))

    # 5. inference
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        face_locator=net.face_locator,
        scheduler=val_noise_scheduler,
        image_proj=net.imageproj,
    )
    pipeline.to(device=device, dtype=weight_dtype)

    audio_emb = process_audio_emb(audio_emb)

    source_image_pixels = source_image_pixels.unsqueeze(0)
    source_image_face_region = source_image_face_region.unsqueeze(0)
    source_image_face_emb = source_image_face_emb.reshape(1, -1)
    source_image_face_emb = torch.tensor(source_image_face_emb)

    source_image_full_mask = [
        (mask.repeat(clip_length, 1))
        for mask in source_image_full_mask
    ]
    source_image_face_mask = [
        (mask.repeat(clip_length, 1))
        for mask in source_image_face_mask
    ]
    source_image_lip_mask = [
        (mask.repeat(clip_length, 1))
        for mask in source_image_lip_mask
    ]


    times = audio_emb.shape[0] // clip_length

    tensor_result = []

    generator = torch.manual_seed(42)

    for t in range(times):

        # Using the profiler to profile the refactored inference step
        
        inference_step(
            t=t,
            clip_length=clip_length,
            audio_emb=audio_emb,
            net=net,
            config=config,
            source_image_pixels=source_image_pixels,
            source_image_face_emb=source_image_face_emb,
            source_image_face_region=source_image_face_region,
            source_image_full_mask=source_image_full_mask,
            source_image_face_mask=source_image_face_mask,
            source_image_lip_mask=source_image_lip_mask,
            img_size=img_size,
            pipeline=pipeline,
            tensor_result=tensor_result,
            generator=generator,
            motion_scale=motion_scale,
            times=times,
        )


        


    tensor_result = torch.cat(tensor_result, dim=2)
    tensor_result = tensor_result.squeeze(0)

    output_file = config.output
    # save the result after all iteration
    tensor_to_video(tensor_result, output_file, driving_audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", default="configs/inference/default.yaml")
    parser.add_argument("--source_image", type=str, required=False,
                        help="source image", default="test_data/source_images/6.jpg")
    parser.add_argument("--driving_audio", type=str, required=False,
                        help="driving audio", default="test_data/driving_audios/singing/sing_4.wav")
    parser.add_argument(
        "--output", type=str, help="output video file name", default=".cache/output.mp4")
    parser.add_argument(
        "--pose_weight", type=float, help="weight of pose", default=1.0)
    parser.add_argument(
        "--face_weight", type=float, help="weight of face", default=1.0)
    parser.add_argument(
        "--lip_weight", type=float, help="weight of lip", default=1.0)
    parser.add_argument(
        "--face_expand_ratio", type=float, help="face region", default=1.2)
    parser.add_argument(
        "--checkpoint", type=str, help="which checkpoint", default=None)
    parser.add_argument("--setting_steps", type=int, default=40)
    parser.add_argument("--setting_cfg", type=float, default=3.5)
    parser.add_argument("--settings_seed", type=int, default=42)
    parser.add_argument("--settings_fps", type=int, default=25)
    parser.add_argument("--settings_motion_pose_scale", type=float, default=1.1)
    parser.add_argument("--settings_motion_face_scale", type=float, default=1.1)
    parser.add_argument("--settings_motion_lip_scale", type=float, default=1.1)
    parser.add_argument("--settings_n_motion_frames", type=int, default=2)
    parser.add_argument("--settings_n_sample_frames", type=int, default=16)

    command_line_args = parser.parse_args()


    inference_process(
        command_line_args,
        command_line_args.setting_steps,
        command_line_args.setting_cfg,
        command_line_args.settings_seed,
        command_line_args.settings_fps,
        command_line_args.settings_motion_pose_scale,
        command_line_args.settings_motion_face_scale,
        command_line_args.settings_motion_lip_scale,
        command_line_args.settings_n_motion_frames,
        command_line_args.settings_n_sample_frames
    )



