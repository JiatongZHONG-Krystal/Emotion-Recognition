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
