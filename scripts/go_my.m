% This script evaluates network inference predictions for DREAM5 Challenge 4
% Silent version - no plots or visual output, only file export
%
% Usage: 
%   go('CLR')  % for CLR technique
%   go('GENIE3_ET')  % for GENIE3_ET technique

function go(technique)
    % If no parameter provided, use CLR as default
    if nargin < 1
        technique = 'CLR';
    end
    
    fprintf('Evaluating technique: %s\n', technique);
    
    % Configuration
    prediction_folder = '/home/jvski/Documents/UTFPR/dupla_diplomacao/classes/thesis/dimreduction/test_matlab_vm2_20250912_mutiples_runs/my_thesis';
    goldfile = '../INPUT/gold_standard_edges_only/DREAM5_NetworkInference_Edges_Network3.tsv';
    pdffile_aupr  = '../INPUT/probability_densities/Network3_AUPR.mat';
    pdffile_auroc = '../INPUT/probability_densities/Network3_AUROC.mat';
    
    % Construct prediction file path
    predictionfile = fullfile(prediction_folder, sprintf('DREAM5_NetworkInference_%s_Network3.tsv', technique));
    
    fprintf('Gold standard file: %s\n', goldfile);
    fprintf('Prediction file: %s\n', predictionfile);
    fprintf('PDF AUPR file: %s\n', pdffile_aupr);
    fprintf('PDF AUROC file: %s\n', pdffile_auroc);
    
    % Check if prediction file exists
    if ~exist(predictionfile, 'file')
        error('Prediction file not found: %s', predictionfile);
    end
    
    % Load data
    gold_edges = load_dream_network(goldfile);
    prediction = load_dream_network(predictionfile);
    pdf_aupr  = load(pdffile_aupr);
    pdf_auroc = load(pdffile_auroc);
    
    % Calculate performance metrics
    [tpr, fpr, prec, rec, L, auroc, aupr, p_auroc, p_aupr] = DREAM5_Challenge4_Evaluation(gold_edges, prediction, pdf_aupr, pdf_auroc);
    
    % Export results
    export_results(technique, tpr, fpr, prec, rec, L, auroc, aupr, p_auroc, p_aupr);
    
    fprintf('Evaluation completed for technique: %s\n', technique);
    fprintf('AUROC: %.4f (p-value: %.4f)\n', auroc, p_auroc);
    fprintf('AUPR:  %.4f (p-value: %.4f)\n', aupr, p_aupr);
end

function export_results(technique, tpr, fpr, prec, rec, L, auroc, aupr, p_auroc, p_aupr)
    % Create output directory if it doesn't exist
    output_dir = 'results';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Export scalar metrics (single values)
    scalar_filename = fullfile(output_dir, sprintf('scalar_metrics_%s.csv', technique));
    scalar_metrics = table(auroc, aupr, p_auroc, p_aupr, ...
        'VariableNames', {'AUROC', 'AUPR', 'P_AUROC', 'P_AUPR'});
    writetable(scalar_metrics, scalar_filename);
    fprintf('Scalar metrics saved to: %s\n', scalar_filename);
    
    % Export curve data (arrays)
    curve_filename = fullfile(output_dir, sprintf('curve_metrics_%s.csv', technique));
    
    % Ensure all arrays have the same length (pad with NaN if necessary)
    max_length = max([length(tpr), length(fpr), length(prec), length(rec), length(L)]);
    
    % Pad arrays to the same length
    tpr_padded = pad_array(tpr, max_length);
    fpr_padded = pad_array(fpr, max_length);
    prec_padded = pad_array(prec, max_length);
    rec_padded = pad_array(rec, max_length);
    L_padded = pad_array(L, max_length);
    
    curve_metrics = table(L_padded', tpr_padded', fpr_padded', prec_padded', rec_padded', ...
        'VariableNames', {'Threshold', 'TPR', 'FPR', 'Precision', 'Recall'});
    writetable(curve_metrics, curve_filename);
    fprintf('Curve metrics saved to: %s\n', curve_filename);
    
    % Export summary for future CSV merging
    export_summary_for_merging(technique, auroc, aupr, p_auroc, p_aupr, length(L));
end

function padded_array = pad_array(arr, target_length)
    % Pad array with NaN to reach target length
    if length(arr) < target_length
        padded_array = [arr; nan(target_length - length(arr), 1)];
    else
        padded_array = arr;
    end
end
function export_summary_for_merging(technique, auroc, aupr, p_auroc, p_aupr, num_thresholds)
    % Export a summary file that can be easily merged with future CSV
    summary_filename = fullfile('results', sprintf('summary_%s.csv', technique));
    
    % FIXED: Use cell array for string data and proper table construction
    dataset_col = {'Network3'};
    network_id_col = 3;
    technique_col = {technique};  % Wrap in cell array
    auroc_col = auroc;
    aupr_col = aupr;
    p_auroc_col = p_auroc;
    p_aupr_col = p_aupr;
    num_thresholds_col = num_thresholds;
    
    % Create table with correct data types
    summary_table = table(...
        dataset_col, ...
        network_id_col, ...
        technique_col, ...
        auroc_col, ...
        aupr_col, ...
        p_auroc_col, ...
        p_aupr_col, ...
        num_thresholds_col, ...
        'VariableNames', {...
        'dataset', ...
        'network_id', ...
        'technique', ...
        'auroc', ...
        'aupr', ...
        'p_auroc', ...
        'p_aupr', ...
        'num_thresholds'});
    
    writetable(summary_table, summary_filename);
    fprintf('Summary for future merging saved to: %s\n', summary_filename);
end