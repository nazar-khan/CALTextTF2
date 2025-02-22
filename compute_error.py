import numpy as np
import editdistance

def cmp_result(label, rec):
    dist_mat = np.zeros((len(label) + 1, len(rec) + 1), dtype='int32')
    dist_mat[0, :] = range(len(rec) + 1)
    dist_mat[:, 0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i, j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i, j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def convert(lst):
    """Convert a list of integers to a single string."""
    return ''.join(map(str, lst))

def process_chr_error(recfile, labelfile):
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    rec_mat = {}
    label_mat = {}

    # Read recognition file
    with open(recfile) as f_rec:
        for line in f_rec:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            rec_mat[key] = latex

    # Read label file
    with open(labelfile) as f_label:
        for line in f_label:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            label_mat[key] = latex

    # Calculate character error rate (CER)
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        dist, llen = cmp_result(label, rec)
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1

    chr_error = float(total_dist) / total_label
    return chr_error

def process_wrd_error(recfile, labelfile, space_ind):
    total_dist = 0
    total_label = 0
    total_line = 0
    rec_mat = {}
    label_mat = {}

    # Read recognition file
    with open(recfile) as f_rec:
        for line in f_rec:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            ss = convert(latex)
            latex = ss.split(str(space_ind))
            rec_mat[key] = latex

    # Read label file
    with open(labelfile) as f_label:
        for line in f_label:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            ss = convert(latex)
            latex = ss.split(str(space_ind))
            label_mat[key] = latex

    # Calculate word error rate (WER)
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        dist, llen = cmp_result(label, rec)
        total_dist += dist
        total_label += llen

    wer = float(total_dist) / total_label
    return wer

def compute_cer_wer(predictions, groundtruths, space_ind):
    #print(len(predictions))
    #print(len(groundtruths))
    #print(predictions)
    #print(groundtruths)
    cer = 0
    wer = 0
    total_words = 0
    for pred, gt in zip(predictions, groundtruths):
        # Compute edit distance (Levenshtein distance) at the character level
        distance = editdistance.eval(pred, gt)
        cer += distance / max(len(gt), 1)  # Avoid division by zero
        # Convert sequences of indices into words
        pred_words = "".join(chr(i) if i != space_ind else " " for i in pred).split()
        gt_words = "".join(chr(i) if i != space_ind else " " for i in gt).split()
        # Compute edit distance (Levenshtein distance) at the word level
        wer += editdistance.eval(pred_words, gt_words)
        total_words += len(gt_words)
    cer = cer / len(predictions)
    wer = wer / total_words if total_words > 0 else 0.0
    return cer, wer
    
#def compute_cer_wer(recfile, labelfile, space_ind):
#    cer = process_chr_error(recfile, labelfile)
#    wer = process_wrd_error(recfile, labelfile, space_ind)
#    return cer, wer

def process(recfile, labelfile, resultfile, space_ind):
    cer = process_chr_error(recfile, labelfile, resultfile)
    wer = process_wrd_error(recfile, labelfile, resultfile, space_ind)
    
    # Save the results to the result file
    with open(resultfile, 'w') as f_result:
        f_result.write(f'CER {cer}\n')
        f_result.write(f'WER {wer}\n')

