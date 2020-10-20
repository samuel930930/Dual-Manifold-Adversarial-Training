python evaluate_classifiers_adv_image_nat.py \
--config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam_AT.yml \
--eval experiments/evaluations/pgd_image_4px.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_adv_image_nat.py \
--config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam_AT.yml \
--eval experiments/evaluations/elastic_05.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_adv_image_nat.py \
--config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam_AT.yml \
--eval experiments/evaluations/fog_128.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_adv_image_nat.py \
--config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam_AT.yml \
--eval experiments/evaluations/gabor_125.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_adv_image_nat.py \
--config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam_AT.yml \
--eval experiments/evaluations/jpeg_16384.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_adv_image_nat.py \
--config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam_AT.yml \
--eval experiments/evaluations/pgd_image_l2_1200.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_adv_image_nat.py \
--config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam_AT.yml \
--eval experiments/evaluations/snow_0625.yml \
--epoch 19 \
--vis


