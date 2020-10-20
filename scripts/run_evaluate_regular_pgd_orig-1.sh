python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_orig_aug_pgd_sgd.yml \
--eval experiments/evaluations/fog_128.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_orig_aug_pgd_sgd.yml \
--eval experiments/evaluations/snow_0625.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_orig_aug_pgd_sgd.yml \
--eval experiments/evaluations/pgd_image_8px.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_orig_aug_pgd_sgd.yml \
--eval experiments/evaluations/gabor_125.yml \
--epoch 19 \
--vis

