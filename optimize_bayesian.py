import numpy as np
from printer import set_up_logging, info
import os
import pickle
from automated_batch_of_simulations import run_batch
from functools import partial
from GPyOpt.methods import BayesianOptimization


def f_only_inittemp(params, quantifier_type, alpha, threshold, num_simulations, run_batch_kwargs):
    info('Starting Bayesian optimization iteration with:', params, quantifier_type, alpha, threshold, num_simulations,
         run_batch_kwargs)
    total_success = run_batch(
        create_plots=False,
        base_seed=0,
        quantifier_type=quantifier_type,
        initial_temperature=float(params[:, 0]),
        threshold=threshold,
        alpha=alpha,
        num_simulations=num_simulations,
        **run_batch_kwargs)
    return -total_success / float(num_simulations)


def f_inittemp_and_alpha(params, quantifier_type, threshold, num_simulations, run_batch_kwargs):
    info('Starting Bayesian optimization iteration with:', params, quantifier_type, threshold, num_simulations,
         run_batch_kwargs)
    total_success = run_batch(
        create_plots=False,
        base_seed=0,
        quantifier_type=quantifier_type,
        initial_temperature=float(params[:, 0]),
        threshold=threshold,
        alpha=float(params[:, 1]),
        num_simulations=num_simulations,
        **run_batch_kwargs)
    return -total_success / float(num_simulations)


def optimize_only_inittemp(quantifier_type, initial_temperature_domain, num_iter_opt_init, num_iter_opt_run, alpha, threshold,
                           num_simulations_in_each_batch, run_batch_kwargs):

    def opt_output_path(path):
        out_dir = os.path.join('opt_bayes', 'opt_only_temperature', quantifier_type)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return os.path.join(out_dir, path)

    info('Starting Bayesian optimization')
    bayes_opt = BayesianOptimization(
        initial_design_numdata=num_iter_opt_init,
        f=partial(f_only_inittemp,
                  quantifier_type=quantifier_type,
                  alpha=alpha,
                  threshold=threshold,
                  num_simulations=num_simulations_in_each_batch,
                  run_batch_kwargs=run_batch_kwargs),
        domain=[
            dict(name='initial_temperature', type='continuous', domain=initial_temperature_domain),
        ])
    bayes_opt.run_optimization(max_iter=num_iter_opt_run,
                               verbosity=True,
                               report_file=opt_output_path('opt_report.txt'),
                               evaluations_file=opt_output_path('opt_evaluations.txt'),
                               models_file=opt_output_path('opt_models.txt'))
    info('Finished Bayesian optimization, saving plots and pickle')
    with open(opt_output_path('bayes_opt.pkl'), 'wb') as f_pkl:
        pickle.dump(bayes_opt, f_pkl)
    bayes_opt.plot_convergence(opt_output_path('opt_convergence.png'))
    bayes_opt.plot_acquisition(opt_output_path('opt_acquisition.png'))


def optimize_inittemp_and_alpha(quantifier_type, alpha_domain, initial_temperature_domain, num_iter_opt_init, num_iter_opt_run, threshold,
                                num_simulations_in_each_batch, run_batch_kwargs):
    def opt_output_path(path):
        out_dir = os.path.join('opt_bayes', 'opt_temperature_and_alpha', quantifier_type)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return os.path.join(out_dir, path)

    info('Starting Bayesian optimization')
    bayes_opt = BayesianOptimization(
        initial_design_numdata=num_iter_opt_init,
        f=partial(f_inittemp_and_alpha,
                  quantifier_type=quantifier_type,
                  threshold=threshold,
                  num_simulations=num_simulations_in_each_batch,
                  run_batch_kwargs=run_batch_kwargs),
        domain=[initial_temperature_domain, alpha_domain])
    bayes_opt.run_optimization(max_iter=num_iter_opt_run,
                               verbosity=True,
                               report_file=opt_output_path('opt_report.txt'),
                               evaluations_file=opt_output_path('opt_evaluations.txt'),
                               models_file=opt_output_path('opt_models.txt'))
    info('Finished Bayesian optimization, saving plots and pickle')
    with open(opt_output_path('bayes_opt.pkl'), 'wb') as f_pkl:
        pickle.dump(bayes_opt, f_pkl)
    bayes_opt.plot_convergence(opt_output_path('opt_convergence.png'))
    bayes_opt.plot_acquisition(opt_output_path('opt_acquisition.png'))


if __name__ == '__main__':
    set_up_logging('out.log')

    # optimize_only_inittemp(
    #     'ALL',
    #     initial_temperature_domain=(1, 2),
    #     num_iter_opt_init=1,
    #     num_iter_opt_run=1,
    #     alpha=0.96,
    #     threshold=1,
    #     num_simulations_in_each_batch=1,
    #     run_batch_kwargs=dict(min_set_size=5, max_set_size=61, number_of_pairs=50))

    # optimize_inittemp_and_alpha(
    #     'ALL',
    #     alpha_domain=(1E-8, 1),
    #     initial_temperature_domain=(1, 2),
    #     num_iter_opt_init=1,
    #     num_iter_opt_run=1,
    #     threshold=1,
    #     num_simulations_in_each_batch=1,
    #     run_batch_kwargs=dict(min_set_size=5, max_set_size=61, number_of_pairs=50))

    optimize_inittemp_and_alpha(
        'ALL',
        alpha_domain=dict(name='alpha', type='discrete', domain=np.arange(0.01, 1.0, 0.01)),
        initial_temperature_domain=dict(name='initial_temperature', type='discrete', domain=np.arange(100, 10000, 100)),
        num_iter_opt_init=10,
        num_iter_opt_run=500,
        threshold=1,
        num_simulations_in_each_batch=100,
        run_batch_kwargs=dict(min_set_size=5, max_set_size=61, number_of_pairs=50))

    optimize_inittemp_and_alpha(
        'NONE',
        alpha_domain=dict(name='alpha', type='discrete', domain=np.arange(0.01, 1.0, 0.01)),
        initial_temperature_domain=dict(name='initial_temperature', type='discrete', domain=np.arange(100, 10000, 100)),
        num_iter_opt_init=10,
        num_iter_opt_run=500,
        threshold=1,
        num_simulations_in_each_batch=100,
        run_batch_kwargs=dict(min_set_size=5, max_set_size=61, number_of_pairs=50))
