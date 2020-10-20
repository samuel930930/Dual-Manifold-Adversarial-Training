python evaluate_classifiers_adv_image.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/fog_128.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_adv_image.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/snow_0625.yml \
--epoch 19 \
--vis && \
python evaluate_classifiers_adv_image.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/gabor_125.yml \
--epoch 19 \
--vis


