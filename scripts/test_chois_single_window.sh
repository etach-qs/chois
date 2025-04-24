python trainer_chois.py \
--window=120 \
--batch_size=32 \
--data_root_folder="/ailab/user/lishujia-hdd/chois_release/data/processed_data" \
--pretrained_model="/ssd1/lishujia/chois_release/pretrained_models/model-10.pt" \
--save_res_folder="/ssd1/lishujia/chois_release/chois_train_set_debug" \
--input_first_human_pose \
--use_random_frame_bps \
--add_language_condition \
--use_object_keypoints \
--add_semantic_contact_labels \
--loss_w_feet=1 \
--loss_w_fk=0.5 \
--loss_w_obj_pts=1 \
--test_sample_res \
--use_guidance_in_denoising \
--test_on_train
#--test_unseen_objects
# --use_object_split