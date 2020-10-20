python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 3 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 7 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 11 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_trades_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 13 \
--vis


