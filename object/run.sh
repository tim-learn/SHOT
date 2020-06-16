~/anaconda3/envs/pytorch/bin/python image_pretrained.py --savename par0.0 --cls_par 0.0 --output seed2020 --seed 2020 --gpu_id 0 --max_epoch 30
~/anaconda3/envs/pytorch/bin/python image_pretrained.py --savename par0.3 --cls_par 0.3 --output seed2020 --seed 2020 --gpu_id 0 --max_epoch 30

~/anaconda3/envs/pytorch/bin/python uda_visda.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --max_epoch 3
~/anaconda3/envs/pytorch/bin/python uda_visda.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --max_epoch 3


~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office --max_epoch 30 --s 2

~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office --max_epoch 30 --s 2

~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office --max_epoch 30 --s 2

~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da pda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da pda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da pda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da pda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da pda --gent '' --threshold 10 --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da pda --gent '' --threshold 10 --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da pda --gent '' --threshold 10 --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da pda --gent '' --threshold 10 --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3_thr10 --cls_par 0.3 --zz val --da pda --gent '' --threshold 10 --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3_thr10 --cls_par 0.3 --zz val --da pda --gent '' --threshold 10 --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3_thr10 --cls_par 0.3 --zz val --da pda --gent '' --threshold 10 --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3_thr10 --cls_par 0.3 --zz val --da pda --gent '' --threshold 10 --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_target_oda.py --savename par0.0 --cls_par 0.0 --zz val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target_oda.py --savename par0.0 --cls_par 0.0 --zz val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target_oda.py --savename par0.0 --cls_par 0.0 --zz val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_target_oda.py --savename par0.0 --cls_par 0.0 --zz val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_target_oda.py --savename par0.3 --cls_par 0.3 --zz val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target_oda.py --savename par0.3 --cls_par 0.3 --zz val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target_oda.py --savename par0.3 --cls_par 0.3 --zz val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_target_oda.py --savename par0.3 --cls_par 0.3 --zz val --da oda --output seed2020 --seed 2020 --gpu_id 0 --dset office-home --max_epoch 30 --s 3



~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --issave 1 --dset office-caltech --net resnet101 --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --issave 1 --dset office-caltech --net resnet101 --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --issave 1 --dset office-caltech --net resnet101 --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --issave 1 --dset office-caltech --net resnet101 --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --issave 1 --dset office-caltech --net resnet101 --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --issave 1 --dset office-caltech --net resnet101 --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --issave 1 --dset office-caltech --net resnet101 --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_target.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --issave 1 --dset office-caltech --net resnet101 --max_epoch 30 --s 3


~/anaconda3/envs/pytorch/bin/python image_multitarget.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_multitarget.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_multitarget.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_multitarget.py --savename par0.0 --cls_par 0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 3

~/anaconda3/envs/pytorch/bin/python image_multitarget.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 0
~/anaconda3/envs/pytorch/bin/python image_multitarget.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 1
~/anaconda3/envs/pytorch/bin/python image_multitarget.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 2
~/anaconda3/envs/pytorch/bin/python image_multitarget.py --savename par0.3 --cls_par 0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --s 3



~/anaconda3/envs/pytorch/bin/python image_multisource.py --savename par0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --t 0
~/anaconda3/envs/pytorch/bin/python image_multisource.py --savename par0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --t 1
~/anaconda3/envs/pytorch/bin/python image_multisource.py --savename par0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --t 2
~/anaconda3/envs/pytorch/bin/python image_multisource.py --savename par0.0 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --t 3

~/anaconda3/envs/pytorch/bin/python image_multisource.py --savename par0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --t 0
~/anaconda3/envs/pytorch/bin/python image_multisource.py --savename par0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --t 1
~/anaconda3/envs/pytorch/bin/python image_multisource.py --savename par0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --t 2
~/anaconda3/envs/pytorch/bin/python image_multisource.py --savename par0.3 --zz val --da uda --output seed2020 --seed 2020 --gpu_id 0 --dset office-caltech --net resnet101 --max_epoch 30 --t 3