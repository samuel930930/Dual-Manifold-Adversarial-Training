python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 13 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_pgd5_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 13 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_manifold_pgd5_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 13 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_normal_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 13 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_manifold_fgsm_sgd.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 13 \
--vis && \
python evaluate_classifiers_adv_latent.py \
--config experiments/classifiers/mixed10_adv_pgd5_adam.yml \
--eval experiments/evaluations/pgd_latent_2px.yml \
--epoch 13 \
--vis


