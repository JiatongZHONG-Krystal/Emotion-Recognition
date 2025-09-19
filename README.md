# Emotion-Recognition-in-Natural-Social-Interactions
Emotion Recognition in Natural Social Interactions: Predicting Pleasantness from Facial Expressions and Body Touch in Movie Scenes

Overview：
This project develops a deep learning-based multimodal system for recognising emotions in naturalistic social interactions. The focus is on predicting the pleasantness of interactions in film scenes, by analysing facial expressions and body movements.
To simulate rich emotional contexts, the dataset is constructed from the TV series Modern Family, which provides diverse interpersonal behaviours such as hugs, pats, hand-holding, and expressive facial cues.

Research Goals：
1.Build a model that predicts the pleasantness of social interactions in cinematic content.
2.Examine the interaction between facial expressions and physical touch in shaping perceived emotions.
3.Compare unimodal vs. multimodal approaches to assess the added value of feature fusion.
4.Relate findings to social perception mechanisms in naturalistic viewing contexts.

Research Questions：
Can deep learning models accurately predict the pleasantness of social interactions based on visual representations of faces and touch in film scenes?
How do different modalities (facial features, touch gestures) interact, and does multimodal fusion improve recognition accuracy?

Here is a description of each file:

1.extract_frames.py：This script processes raw video files and extracts every frame as an image. It supports multiple video formats and organizes the extracted frames into subfolders named after the original video file, making the data ready for further analysis.

2.track_main_subject.py：This script detects and tracks faces across frames using MTCNN and DeepSort, calculates motion energy to identify the most active participants, and crops their faces into separate folders for each track. It ensures that only the main interacting subjects are retained for analysis.

3.batch_track_active.py：This script automates the process of selecting and cropping active face tracks for all videos. It iterates through the extracted frame folders, and saves the top-K most active face tracks into structured output folders. If results already exist for a video, it skips reprocessing, ensuring efficient batch execution across the entire dataset.

4.pipeline_main_emonet.py：This script builds an end-to-end pipeline that tracks the main face in each video and extracts emotional features using EmoNet. For each video, it first checks if frames are available, then crops the main subject’s face with track_main_subject.py, and finally runs batch_predict_emonet.py to generate EmoNet features. The results are saved in organized folders for frames, cropped faces, and extracted features, making the pipeline fully automated.

5.batch_predict_active.py：This script runs EmoNet predictions on the cropped active face tracks. For each video, it iterates over the tracked face folders (track_*), calls batch_predict_emonet.py to extract emotion features, and saves the results (including emonet.json) into structured output directories. It also skips tracks that have already been processed, ensuring efficient batch execution across the dataset.

6.batch_predict_emonet.py：This script applies EmoNet to cropped face images to predict emotional dimensions. It detects faces in each image, preprocesses them, and runs EmoNet to output valence and arousal scores. Results are averaged across all frames in a folder and saved as emonet.json, providing a compact emotional representation for each track or subject.

7.collect_active_results.py：This script gathers all EmoNet prediction results from active face tracks and merges them into a single CSV file. It reads each emonet.json output, appends metadata such as video name and track ID, and consolidates everything into all_emonet_active.csv, providing a structured dataset for further statistical analysis or model training.

8.clean_emonet_table.py：This script cleans and filters the aggregated EmoNet results. It removes tracks with too few frames, optionally drops rows with missing valence/arousal values, and keeps only the top-N longest tracks per video. The cleaned data is saved as a new CSV file, ensuring a high-quality and consistent dataset for downstream analysis.

9.batch_predict_openpose.py：This script batch-processes all videos with OpenPose to extract body pose information. For each video, it generates both a set of JSON files containing keypoint coordinates and a rendered video with pose overlays. The script automatically skips already-processed files, cleans up old folders if necessary, and organizes outputs into structured directories. It provides an efficient way to run OpenPose on an entire dataset of videos.

10.check_openpose_outputs.py：This script identifies videos where OpenPose processing failed and automatically retries them. It first checks each video’s output folder to verify that both JSON keypoint files and the rendered video exist. Failed cases are logged, and for those with spaces in filenames, the script renames them to use underscores and reruns OpenPose only on the failed videos. This ensures data completeness while avoiding unnecessary reprocessing.

11.collect_openpose_features_person12_std.py：This script processes OpenPose JSON outputs to extract stable body pose features for up to two main persons in each video. It uses bounding box size and center matching to maintain consistent identities across frames, computes mean and standard deviation of BODY_25 keypoints, and filters out low-quality tracks with fewer than 10 frames. The results provide structured pose features for downstream analysis.

12.merge_emonet_openpose.py：This script merges EmoNet features (facial emotion predictions) with OpenPose features (body pose statistics) into a unified dataset. It filters out short tracks, keeps the top-2 tracks per video, converts EmoNet results into a wide format with person_1 and person_2, and joins them with corresponding OpenPose features. The script also attaches touch-related labels from the annotation file. Two outputs are generated: one with all merged samples, and another with only videos where both persons have valid data (has_both_both).

13.train_baseline_early_fusion.py：This script trains and evaluates a baseline emotion recognition model using early fusion of EmoNet and OpenPose features. It applies a multi-layer perceptron (MLP) classifier with standardized inputs and performs 5-fold stratified cross-validation. The script computes accuracy, macro-F1 scores, and averaged confusion matrices across folds. Results (metrics, confusion matrix, and the trained pipeline) are saved for reproducibility.

14.train_single_modal_models.py：This script evaluates single-modality baselines using either EmoNet features or OpenPose features independently. It trains a multi-layer perceptron (MLP) classifier with standardized inputs and performs 5-fold stratified cross-validation. Results include accuracy and macro-F1 scores for both modalities, which are saved separately along with the corresponding trained pipelines.
