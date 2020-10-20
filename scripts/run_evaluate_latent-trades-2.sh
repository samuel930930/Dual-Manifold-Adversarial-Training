python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 15 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 17 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 19 \
--vis


