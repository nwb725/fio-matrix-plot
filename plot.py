#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib.ticker import AutoMinorLocator
from binary import BinaryUnits, DecimalUnits, convert_units
from pathlib import Path
import re
import datetime
import glob
import json
import argparse
import tomllib

colors = [
    '#72AB97',
    '#728CA6',
    '#FFDCAA',
    '#FFC6AA',
]

def indexes():
    return ['config','bs','qd','jobcount','workload']

def indexes_no_config():
    i = indexes()
    i.remove('config')
    return i

def get_columns(index):
    columns = set(indexes())
    return columns - set([index]) - set(['config'])

def load_log(path):
    glob_path = f'{path}/log-*.log'
    try:
        log_path = glob.glob(glob_path)[0]
    except IndexError as e:
        print(f"Could not find log file at {glob_path}")
        raise e

    with open(log_path, "rt") as f:
        log_data = f.read()

    return log_data

def get_ip_from_log(log_data):
    match = re.search(r"\d+\.\d+\.\d+\.\d+", log_data)
    if not match:
        return "No IP"
    ip = match.group(0)
    return ip

def get_lang_from_log(log_data):
    match = re.search(r"null_blk", log_data)
    if match:
        return "c"
    else:
        return "rust"

def load_file(path, config):
    files = glob.glob(f'{path}/**/*.json')
    frame = pd.DataFrame()

    log_data = load_log(f'{path}')
    ip = get_ip_from_log(log_data)
    lang = get_lang_from_log(log_data)

    for jfile in files:
        with open(jfile, "rt") as f:
            data = json.load(f)

        timestamp = datetime.datetime.fromtimestamp(data['timestamp'])

        job = data['jobs'][0]
        options = job['job options']
        
        
        if options['bs'].endswith('m'):
            bs = int(options['bs'].rstrip('m')) * 1024 * 1024
        elif options['bs'].endswith('k'):
            bs = int(options['bs'].rstrip('k')) * 1024
        else:
            bs = int(options['bs'])

        bs = format_bs(bs, 'bs')
        qd = int(options['iodepth'])
        iops = int(data["jobs"][0]["read"]["iops"]) + int(data["jobs"][0]["write"]["iops"])
        jobcount = int(options['numjobs'])
        workload = options['rw']

        new = pd.DataFrame({
            'config': config,
            'lang': lang,
            'qd': qd,
            'bs': bs,
            'jobcount': jobcount,
            'workload': workload,
            'iops': iops,
            'ip': ip,
            'timestamp': timestamp,
        }, index=[0])
        frame = pd.concat([frame, new], ignore_index=True)
    return frame

def append_single(frame, path, config):
    frame = pd.concat([frame, load_file(path, config)], ignore_index=True)
    return frame

def calculate_difference(frame, a, b, data_conf):
    group = frame\
        .groupby(indexes())['iops']

    stat = pd.DataFrame({
        "samples": group.count(),
        "mean": group.mean(),
        "variance": group.var(),
        "stddev": group.std(),
    }).reset_index().pivot(index=indexes_no_config(), columns=['config']).sort_index(level=['qd'])

    # Only keeping data specified in config.
    stat = stat[stat.index.get_level_values('qd').isin(data_conf['qd'])]
    stat = stat[stat.index.get_level_values('bs').isin(data_conf['bs'])]
    stat = stat[stat.index.get_level_values('workload').isin(data_conf['workload'])]
    stat = stat[stat.index.get_level_values('jobcount').isin(data_conf['jobcount'])]

    confidence = 95
    tval = stat['samples'][a].map(lambda x: np.abs(sp.stats.t.ppf((100-confidence) / 200, x)))
    stderr = ( (stat['variance'][a] / stat['samples'][a]) + (stat['variance'][b] / stat['samples'][b]) ).apply(np.sqrt)
    interval = stderr * tval

    tval_a = stat['samples'][a].map(lambda x: np.abs(sp.stats.t.ppf((100-confidence) / 200, x)))
    stderr_a = stat['variance'][a] / stat['samples'][a].apply(np.sqrt)
    interval_a = stderr * tval

    tval_b = stat['samples'][b].map(lambda x: np.abs(sp.stats.t.ppf((100-confidence) / 200, x)))
    stderr_b = stat['variance'][b] / stat['samples'][b].apply(np.sqrt)
    interval_b = stderr * tval

    result = pd.DataFrame({
        'diff': stat['mean'][a] - stat['mean'][b],
        'diff_interval': interval,
        'relative_diff': (stat['mean'][a] - stat['mean'][b]) / stat['mean'][b],
        'relative_diff_interval': interval / stat['mean'][b],
        a: stat['mean'][a],
        b: stat['mean'][b],
        f'{a}_interval': interval_a,
        f'{b}_interval': interval_b,
        f'{a}_samples': stat['samples'][a],
        f'{b}_samples': stat['samples'][b],
    })
    return result

def generate_query_string(query):
    query_components = list()
    for key,value in query.items():
        query_components.append(f'{key} == {repr(value)}')

    query = ' and '.join(query_components)
    return query

def format_bs(bs, kind):
    if kind == 'bs':
        # This is scuffed
        bs = parse_size(str(bs))        
        (bs,unit) = convert_units(bs)
        return f'{bs:.0f}{unit}'
    else:
        return bs
    


def plot(axes, result, field, query, index):
    data = result[[field, f"{field}_interval"]]\
        .reset_index()\
        .query(generate_query_string(query))\
        .pivot(index=[index], columns=get_columns(index))

    # Sorts the subplot xaxis, important because block sizes have units.
    # Fails if subplot xaxis is workload.
    if index != 'workload':
        data = data.sort_index(key=lambda idx: idx.map(parse_size))

    ax = data[field]\
        .plot.bar(ax=axes, yerr=data[f"{field}_interval"], capsize=1.5, error_kw={'elinewidth':0.5}, edgecolor='black', lw=0.5, color=colors)
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([format_bs(x, index) for x in data.index]))
    ax.axhline(0, color='black', lw=0.5, label='_nolegend_')
    #ax.set_title(f"qd {qd}, {workload}")
    #ax.set_xlabel(f"Queue Depth {qd}")
    ax.set_xlabel("")
    #ax.set_ylabel(f"{workload}")
    #ax.legend([workload], loc='best')
    ax.legend().remove()
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='both')
    #ax.xaxis.set_label_position('top')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

def plot_throughput(axes, frame, base, new):
    result = calculate_difference(frame, new, base)

    data = result[[base, new]]\
        .reset_index()\
        .pivot(index=['bs'], columns=get_columns('bs'))\
        .sort_index(axis=1, level='qd')

    error = result[[f'{base}_interval', f'{new}_interval']]\
        .reset_index()\
        .pivot(index=['bs'], columns=get_columns('bs'))\
        .sort_index(axis=1, level='qd').to_numpy().transpose()

    ax = data \
        .plot\
        .bar(
            ax=axes,
            #yerr=error,
            logy=True,
            capsize=1.5, error_kw={'elinewidth':0.5}, edgecolor='black', lw=0.5,
            color=colors
        )
    bars = ax.patches
    groups = len(data.index)
    for (bar, hatch) in zip(bars, [None]*groups*4 + ['//']*groups*4):
        if hatch != None:
            bar.set_hatch(hatch)

    ax.xaxis.set_major_formatter(ticker.FixedFormatter([format_bs(x) for x in data.index]))
    ax.axhline(0, color='black', lw=0.5, label='_nolegend_')
    ax.set_xlabel("")
    ax.legend().remove()
    ax.set_axisbelow(True)
    #ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='both')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

def map_config_axis_names(config):
    mapping = { 'jobcounts': 'jobcount', 'queue_depths': 'qd', 'block_sizes': 'bs', 'workloads': 'workload' }
    return mapping[config]

def axis_name_mapping(config_item):
    return {
        'block_sizes': 'Block Sizes',
        'queue_depths': 'qd',
        'jobcounts': 'Threads (cores)',
        'workloads': '',
        }[config_item]

# This gets send to 'calculate_difference' to remove data points that is _not_ specified in
# the configuration.
def build_plot_dict(config):
    return {
            map_config_axis_names(config['plot_gridy']): config[config['plot_gridy']],
            map_config_axis_names(config['plot_gridx']): config[config['plot_gridx']],
            map_config_axis_names(config['subplotx']): config[config['subplotx']],
            map_config_axis_names(config['barcluster']): config[config['barcluster']]
           }

def parse_size(s):
    if s.endswith("KiB"):
        return int(s[:-3]) * 1024
    elif s.endswith("MiB"):
        return int(s[:-3]) * 1024 * 1024
    elif s.endswith("B"):
        return int(s[:-1])
    else:
        return int(s)

def plot_rnull(frame, field, config, base = None, new = None, title = 'Comparison'):
    axis_conf = build_plot_dict(config)    
    result = calculate_difference(frame, new, base, axis_conf)

    plotgridy = config[config['plot_gridy']]
    plotgridx = config[config['plot_gridx']]
    subplotx = config[config['subplotx']]
    barcluster = config[config['barcluster']]
    
    fig, axes = plt.subplots(len(plotgridy), len(plotgridx), sharey=True, sharex=True, figsize=(13,6), squeeze=False)
    fig.suptitle(title)
    for i, pgy in enumerate(plotgridy):
        for j, pgx in enumerate(plotgridx):
            gy, gx, sp = map_config_axis_names(config['plot_gridy']), map_config_axis_names(config['plot_gridx']), map_config_axis_names(config['subplotx'])
            plot(axes[i][j], result, field, {gy: pgy, gx: pgx}, sp)
            
    plotgridy_axis_name = axis_name_mapping(config['plot_gridy'])
    for i, pgy in enumerate(plotgridy):
        axes[i][0].set_ylabel(f"{plotgridy_axis_name} {pgy}")

    plotgridx_axis_name = axis_name_mapping(config['plot_gridx'])
    for j, pgx in enumerate(plotgridx):
        axes[0][j].set_title(f"{plotgridx_axis_name} {pgx}")
    
    fig.text(0.01, 0.5, 'IO/s Difference Relative', va='center', rotation='vertical')
    fig.text(0.5, 0.01, axis_name_mapping(config['subplotx']), ha='center')
    fig.legend(barcluster, loc='lower left', ncols=3, title=axis_name_mapping(config['barcluster']), bbox_to_anchor=(0.03,0.85))    
    fig.tight_layout(pad=1)
    plt.subplots_adjust(left=0.08, top=0.8)

    print("Mean of difference: {:.3}".format(result[field].mean()))
    print("Samples {}: {:.3}".format(base, result[f'{base}_samples'].mean()))
    print("Samples {}: {:.3}".format(new, result[f'{new}_samples'].mean()))

def violin(ax, frame, base, new, workload, config):
    axis_conf = {
        'plot_gridy': config[config['plot_gridy']],
        'plot_gridx': config[config['plot_gridx']],
        'subplotx': config[config['subplotx']],
        'barcluster': config[config['barcluster']]
    } 
    pgx, pgy, sp, bc = axis_conf['plot_gridx'], axis_conf['plot_gridy'], axis_conf['subplotx'], axis_conf['barcluster']

    # This must be generated by 1..len('barcluster'), 1+len('barcluster')..
    # repeated len('subplotx') times
    vlines = [(i + 1) * (len(bc) + 1) for i in range(len(sp)-1)]
    positions = [i * (len(bc) + 1) + j + 1 for i in range(len(sp)) for j in range(len(bc))] 
    
    query = {map_config_axis_names(config['plot_gridy']):  pgy, map_config_axis_names(config['plot_gridx']): pgx, 'config': base}
    frame_c = frame.query(generate_query_string(query))
    query = {map_config_axis_names(config['plot_gridy']):  pgy, map_config_axis_names(config['plot_gridx']): pgx, 'config': new}
    frame_r = frame.query(generate_query_string(query))
    data_c = list()
    data_r = list()
    for sbx in sp:
        for bcs in bc:
            query = {map_config_axis_names(config['subplotx']): sbx, map_config_axis_names(config['barcluster']): bcs}
            col_c = frame_c.query(generate_query_string(query))['iops']
            col_r = frame_r.query(generate_query_string(query))['iops']
            mean = col_c.mean()
            col_c /= mean
            col_r /= mean
            data_c.append(col_c)
            data_r.append(col_r)
    side = 'both'
    width = 0.8
    parts_c = ax.violinplot(data_c, side='low', showextrema=True, widths=width, positions=positions)
    parts_r = ax.violinplot(data_r, side='high', showextrema=True, widths=width, positions=positions)

    for i,pc in enumerate(parts_c['bodies']):
        pc.set_facecolor(colors[i%3])
        pc.set_edgecolor(colors[i%3])
        #pc.set_linewidth(2)

    for i,pc in enumerate(parts_r['bodies']):
        pc.set_facecolor(colors[i%3])
        pc.set_edgecolor(colors[i%3])
        #pc.set_linewidth(2)
        pc.set_hatch('X+*')
        pc.set_alpha(0.5)

    parts_c['cbars'].set_linewidth(0.5)
    parts_c['cmaxes'].set_linewidth(0.5)
    parts_c['cmins'].set_linewidth(0.5)
    parts_r['cbars'].set_linewidth(0.5)
    parts_r['cmaxes'].set_linewidth(0.5)
    parts_r['cmins'].set_linewidth(0.5)

    ax.vlines(vlines, 0, 1,  transform=ax.get_xaxis_transform())

    ax.xaxis.set_major_locator(ticker.FixedLocator([2,6,10,14,18]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([format_bs(x, map_config_axis_names(config['subplotx'])) for x in sp]))
    ax.set_xlabel("")
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    #ax.yaxis.grid(True, which='both')
    #ax.xaxis.grid(True, which='both')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

def plot_null_violin(frame, config, base = None, new = None, title = 'Normalized Density'):
    axis_conf = {
        'plot_gridy': config[config['plot_gridy']],
        'plot_gridx': config[config['plot_gridx']],
        'subplotx': config[config['subplotx']],
        'barcluster': config[config['barcluster']]
    }    

    plotgridy = config[config['plot_gridy']]
    plotgridx = config[config['plot_gridx']]
    subplotx = config[config['subplotx']]
    barcluster = config[config['barcluster']]
    
    fig, axes = plt.subplots(len(plotgridy), len(plotgridx), sharey=True, sharex=True, figsize=(13,6), squeeze=False)
    fig.suptitle(title)

    for i, pgy in enumerate(plotgridy):
        for j, pgx in enumerate(plotgridx):
            gy, gx, sp = map_config_axis_names(config['plot_gridy']), map_config_axis_names(config['plot_gridx']), map_config_axis_names(config['subplotx'])
            violin(axes[i][j], frame, base, new, config['plot_gridy'], config)
            

    

    plotgridy_axis_name = axis_name_mapping(config['plot_gridy'])
    for i, pgy in enumerate(plotgridy):
        axes[i][0].set_ylabel(f"{plotgridy_axis_name} {pgy}")

    plotgridx_axis_name = axis_name_mapping(config['plot_gridx'])
    for j, pgx in enumerate(plotgridx):
        axes[0][j].set_title(f"{plotgridx_axis_name} {pgx}")
        
    fig.text(0.01, 0.5, 'IO/s Difference Relative', va='center', rotation='vertical')
    fig.text(0.5, 0.01, axis_name_mapping(config['subplotx']), ha='center')
    fig.legend(barcluster, loc='lower left', ncols=3, title=axis_name_mapping(config['barcluster']), bbox_to_anchor=(0.03,0.85)) 


def null_cli(path_a, path_b, name_a, name_b, out_path, out_name, config):
    frame = pd.DataFrame()
    frame = append_single(frame, path_a, name_a)
    frame = append_single(frame, path_b, name_b)

    plot_rnull(frame, 'relative_diff', config, base = name_a, new=name_b, title=r"Throughput (Bare Metal)")
    plt.savefig(f'{out_path}/{out_name}.svg')
    plot_null_violin(frame, config, name_a, name_b)
    plt.savefig(f'{out_path}/{out_name}-density.svg')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-a", required=True)
    parser.add_argument("--name-a", default="a")
    parser.add_argument("--path-b", required=True)
    parser.add_argument("--name-b", default="b")
    parser.add_argument("--out-path", default=".")
    parser.add_argument("--out-name", default="plot")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    conf = tomllib.load(open(args.config, "rb"))
     
    null_cli(args.path_a, args.path_b, args.name_a, args.name_b, args.out_path, args.out_name, conf)
       
if __name__ == "__main__":
    main()
