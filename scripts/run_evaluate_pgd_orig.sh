python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd.yml \
--eval experiments/evaluations/pgd_image_8px.yml \
--epoch 19 && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_adv_pgd5_sgd.yml \
--eval experiments/evaluations/pgd_image_8px.yml \
--epoch 19 && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_manifold_pgd5_sgd.yml \
--eval experiments/evaluations/pgd_image_8px.yml \
--epoch 19 && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_normal_sgd.yml \
--eval experiments/evaluations/pgd_image_8px.yml \
--epoch 19 && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_manifold_fgsm_sgd.yml \
--eval experiments/evaluations/pgd_image_8px.yml \
--epoch 19 && \
python evaluate_classifiers_orig_image.py \
--config experiments/classifiers/mixed10_adv_pgd5_adam.yml \
--eval experiments/evaluations/pgd_image_8px.yml \
--epoch 19


