import pandas as pd
from glob import glob
import warnings
import numpy as np

def activity_to_number(row):
    activity_map = {"Y":1, "N":0}
    activity = row["activity"]
    if pd.isna(activity):
        warnings.warn(f"{row['id']} {row['family']} {row['model']} {row['round']}: activity is null")
    else:
        return activity_map[activity]


def read_metrics_csvs(directory):
    
    metric_tables = dict()
    merged_data = pd.DataFrame(columns=['id', 'family', 'model', 'round'])
    for dataset_i in glob(directory+"/*.csv"):
        filename = dataset_i.split("/")[-1].split(".")[0]
        family, exp_round, model, metric = filename.split("_",3)
        
        df = pd.read_csv(dataset_i, index_col=None, dtype={'id': object})
        df["family"] = family
        df["model"] = model
        df["round"] = exp_round
        
        if (metric) not in metric_tables:
            metric_tables[metric] = list() #pd.DataFrame(columns=['id', 'family', 'model', 'round'])
        metric_tables[metric].append(df)

    for metric in metric_tables:
        metric_tables[metric] = pd.concat(metric_tables[metric], ignore_index=True)
        merged_data = pd.merge(merged_data, metric_tables[metric], how='outer', on=['id', 'family', 'model', 'round'])

    #merged_data.to_csv("test_out.csv")
    if "activity" in merged_data.columns:
        merged_data["activity"] = merged_data.apply(activity_to_number, axis=1)
    if "hamming" in merged_data.columns:
        merged_data["identity"] = 1.0 - merged_data["hamming"]

    return merged_data

def read_fastas(directory):
    df_dict = {"id": list(), "family": list(), "model": list(), "round": list(), "sequence": list()}
    for dataset_i in glob(directory+"/*.fasta"):
        filename = dataset_i.split("/")[-1].split(".")[0]
        family, exp_round, model = filename.split("_",3)
        (names, sequences) = parse_fasta(dataset_i, return_names=True)
        df_dict["id"] += names
        df_dict["family"] += [family] * len(names)
        df_dict["model"] += [model] * len(names)
        df_dict["round"] += [exp_round] * len(names)
        df_dict["sequence"] += sequences
    return pd.DataFrame(df_dict)


def precision_curve(x,y, percentile = False):
    """
        x: a list of reals
        y: a list of bools (Positive vs Negative)
        percentile: if True then the first column of output is percentile otherwise it is x

        returns:
            a numpy array with two columns, x (or percentile x), and cumulative positive rate up to x.

    """
#    x_precentile = [percentileofscore(x,z) for z in x] #probably not very efficient because it has to sort x each time, later versions of scipy stats can do it in one call.
    arr = np.array([x,y]).T
    arr = arr[arr[:,0].argsort()]
    arr_len = arr.shape[0]
    cumulative_precision = np.zeros((arr_len))

    positives = 0 
    total = 0
    
    for i in range(arr_len):
        total += 1
        if arr[i, 1]:
            positives += 1
        cumulative_precision[i] = positives/total

    if percentile:
        percentile_vals = np.array(range(arr_len)).T / arr_len
        out = np.hstack( (percentile_vals, cumulative_precision) )
    else:
        out = np.hstack( (arr[:,0], cumulative_precision) )

    return out
    

def _open_if_is_name(filename_or_handle, mode="r"):
    """
        if a file handle is passed, return the file handle
        if a Path object or path string is passed, open and return a file handle to the file.
        returns:
            file_handle, input_type ("name" | "handle")
    """
    out = filename_or_handle
    input_type = "handle"
    try:
        out = open(filename_or_handle, mode)
        input_type = "name"
    except TypeError:
        pass
    except Exception as e:
        raise(e)

    return (out, input_type)

def parse_fasta(filename, return_names=False, clean=None, full_name=False): 
    """
        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        
        input:
            filename: the name of a fasta file or a filehandle to a fasta file.
            return_names: if True then return two lists: (names, sequences), otherwise just return list of sequences
            clean: {None, 'upper', 'delete', 'unalign'}
                    if 'delete' then delete all lowercase "." and "*" characters. This is usually if the input is an a2m file and you don't want to preserve the original length.
                    if 'upper' then delete "*" characters, convert lowercase to upper case, and "." to "-"
                    if 'unalign' then convert to upper, delete ".", "*", "-"
            full_name: if True, then returns the entire name. By default only the part before the first whitespace is returned.
        output: sequences or (names, sequences)
    """
    
    prev_len = 0
    prev_name = None
    prev_seq = ""
    out_seqs = list()
    out_names = list()
    (input_handle, input_type) = _open_if_is_name(filename)

    for line in input_handle:
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == ">":
            if full_name:
                name = line[1:]
            else:
                parts = line.split(None, 1)
                name = parts[0][1:]
            out_names.append(name)
            if (prev_name is not None):
                out_seqs.append(prev_seq)
            prev_len = 0
            prev_name = name
            prev_seq = ""
        else:
            prev_len += len(line)
            prev_seq += line
    if (prev_name != None):
        out_seqs.append(prev_seq)

    if input_type == "name":
        input_handle.close()
    
    if clean == 'delete':
        # uses code from: https://github.com/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        translation = str.maketrans(deletekeys)
        remove_insertions = lambda x: x.translate(translation)

        for i in range(len(out_seqs)):
            out_seqs[i] = remove_insertions(out_seqs[i])
    
    elif clean == 'upper':
        deletekeys = {'*': None, ".": "-"}
        translation = str.maketrans(deletekeys)
        remove_insertions = lambda x: x.translate(translation)

        for i in range(len(out_seqs)):
            out_seqs[i] = remove_insertions(out_seqs[i].upper())
    elif clean == 'unalign':
        deletekeys = {'*': None, ".": None, "-": None}
        
        translation = str.maketrans(deletekeys)
        remove_insertions = lambda x: x.translate(translation)
        
        for i in range(len(out_seqs)):
            out_seqs[i] = remove_insertions(out_seqs[i].upper())
    elif clean is not None:
        raise ValueError(f"unrecognized input for clean parameter: {clean}")

    if return_names:
        return out_names, out_seqs
    else:
        return out_seqs
