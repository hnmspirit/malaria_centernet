python main.py ctdet --exp_id="res18prm" --arch="res18prm" \
--gpus=0 --num_workers=2 \
--reg_loss="sl1" --cat_spec_wh \
--num_epochs=20 --batch_size=16 --lr_step="45,65"