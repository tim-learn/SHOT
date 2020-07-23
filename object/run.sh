# Table 3        A->D,W
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/

# Table 4        A->C,P,R
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --cls_par 0.3 --da uda --dset office-home --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/

# Table 5        VisDA-C
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset VISDA-C --net resnet101 --lr 1e-3 --max_epoch 10 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --cls_par 0.3 --da uda --dset VISDA-C --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --net resnet101 --lr 1e-3

# Table 7        A->C,P,R (PDA)
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --output ckps/source/ --da pda --gpu_id 0 --dset office-home --max_epoch 50 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --cls_par 0.3 --threshold 10 --da pda --dset office-home --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/

# Table 7        A->C,P,R (ODA)
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office-home --max_epoch 50 --s 0
~/anaconda3/envs/pytorch/bin/python image_target_oda.py --cls_par 0.3 --da oda --dset office-home --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/


# Table 8        C,D,W->A (MSDA)
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 100 --s 1
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 100 --s 2
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 100 --s 3

~/anaconda3/envs/pytorch/bin/python image_target.py --cls_par 0.3 --da uda --dset office-caltech --net resnet101 --gpu_id 0 --s 1 --output_src ckps/source/ --output ckps/target/
~/anaconda3/envs/pytorch/bin/python image_target.py --cls_par 0.3 --da uda --dset office-caltech --net resnet101 --gpu_id 0 --s 2 --output_src ckps/source/ --output ckps/target/
~/anaconda3/envs/pytorch/bin/python image_target.py --cls_par 0.3 --da uda --dset office-caltech --net resnet101 --gpu_id 0 --s 3 --output_src ckps/source/ --output ckps/target/

~/anaconda3/envs/pytorch/bin/python image_multisource.py --cls_par 0.3 --da uda --dset office-caltech --gpu_id 0 --t 0 --output_src ckps/source/ --output ckps/target/

# Table 8        A->(C,D,W)(MTDA)
~/anaconda3/envs/pytorch/bin/python image_multitarget.py --cls_par 0.3 --da uda --dset office-caltech --net resnet101 --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/


# Table 9       ImageNet->Caltech(PDA)
~/anaconda3/envs/pytorch/bin/python image_pretrained.py --gpu_id 0 --output ckps/target --cls_par 0.3

