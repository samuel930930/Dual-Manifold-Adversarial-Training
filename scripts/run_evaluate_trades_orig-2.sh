python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/pgd_image_l2_1200.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/elastic_05.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/jpeg_16384.yml \
--epoch 19 \
--vis


