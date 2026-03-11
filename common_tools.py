#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost as xgb
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Constants
yields_BKGoverSIG = 1896
FEATURES = [
    "Muons_PT_Lead", "Muons_PT_Sub", "Event_VT_over_HT", "dR_mu0_mu1", "Jets_jetMultip", "Event_MET","Muons_CosThetaStar","Event_MET_Sig","DPHI_MET_DIMU"
]
labels={"Muons_PT_Lead":r"$p_{T}^{\mu_1}$", "Muons_PT_Sub":r"$p_{T}^{\mu_2}$", "Event_VT_over_HT":r"$V_{T}/H_{T}$", "DPHI_MET_DIMU":r"$\Delta\phi (\mu\mu, E_{T}^{miss})$" ,
        "dR_mu0_mu1":r"$\Delta R(\mu_1, \mu_2)$", "Jets_jetMultip":r"$N_{j}$", "Event_MET":r"$E_{T}^{miss}$","Muons_CosThetaStar":r"$cos \theta^{*}$","Event_MET_Sig":r"$\sigma (E_{T}^{miss}$)"}

def plot_test_train_distributions(X_train, y_train, w_train, X_valid, y_valid, w_valid, fold="",title=""):
    # 1️⃣ **Feature Distributions**
    print(f"vh2lep_BDT_features_fold{fold}.pdf")
    Signal_train = X_train[y_train==1]
    Signal_test  = X_valid[y_valid==1]
    Signal_weight_train = w_train[y_train==1]
    Signal_weight_test  =   w_valid[y_valid==1]
    print(Signal_train.shape, Signal_weight_train.shape)
    print(Signal_test.shape, Signal_weight_test.shape)

    Bkg_train    = X_train[y_train==0]
    Bkg_test     = X_valid[y_valid==0]
    Bkg_weight_train =  w_train[y_train==0] 
    Bkg_weight_test  =  w_valid[y_valid==0]

    num_features = len(FEATURES)
    plt.figure(figsize=(15, num_features * 5))  # Adjust figure size based on number of features
    plt.suptitle(title,fontsize=14, y=1.0)

    for i, feature in enumerate(FEATURES):
        plt.subplot(num_features, 2, i + 1)
        #sns.histplot(Signal_train[feature].values,  weights= Signal_weight_train.values, bins=50, color="r", label="Signal", kde=True, stat="probability", alpha=0.5)
        bin_counts, bin_edges = np.histogram(Signal_train[feature], weights=Signal_weight_train, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
        plt.bar(bin_centers, bin_counts/np.sum(bin_counts), width=np.diff(bin_edges), 
        align='center', color="red", alpha=0.5, label="Signal (train)")        

        #sns.histplot(Bkg_train[feature].values, weights=Bkg_weight_train.values, bins=50, color="b", label="Background", kde=True, stat="probability", alpha=0.5)
        bin_counts, bin_edges = np.histogram(Bkg_train[feature], weights=Bkg_weight_train, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
        plt.bar(bin_centers, bin_counts/np.sum(bin_counts), width=np.diff(bin_edges), 
        align='center', color="blue", alpha=0.5, label="Background (train)")       

        bin_counts, bin_edges = np.histogram(Signal_test[feature], weights=Signal_weight_test, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
        plt.scatter(bin_centers, bin_counts/np.sum(bin_counts), color="red", marker='o', s=40, label="Signal (test)")

        bin_counts_b, bin_edges_b = np.histogram(Bkg_test[feature], weights=Bkg_weight_test, bins=50)
        bin_centers_b = (bin_edges_b[:-1] + bin_edges_b[1:]) / 2  # Midpoints of bins
        plt.scatter(bin_centers_b, bin_counts_b/np.sum(bin_counts_b), color="blue", marker='o', s=40, label="Background (test)")

        plt.xlabel(labels[feature])
        plt.ylabel("Probability")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"vh2lep_BDT_features_fold{fold}.png", format="png")
    plt.show()



def get_opt_parameters(X_tr, y_tr, w_tr, num_iter=40,random_seed=42):
    np.random.seed(random_seed+40)
    model = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.01,use_label_encoder=False, eval_metric='logloss', scale_pos_weight=yields_BKGoverSIG , verbosity=0)
    # Define parameter grid with distributions
    param_dist = {
    'max_depth': np.random.randint(3, 10, 5),
    'learning_rate': np.random.uniform(0.01, 0.3, 5),
    'n_estimators': np.random.randint(100, 500, 5),
    'subsample': np.random.uniform(0.6, 1.0, 5),
    'colsample_bytree': np.random.uniform(0.6, 1.0, 5),
    'reg_alpha': np.random.uniform(0, 1, 5),
    'reg_lambda': np.random.uniform(0, 1, 5)
    }

    # Perform randomized search
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=num_iter, scoring='neg_log_loss', cv=3, verbose=1, random_state=random_seed)
    random_search.fit(X_tr , y_tr ,sample_weight=w_tr )

    # Best parameters
    print("Best parameters:", random_search.best_params_)
    best_params=  random_search.best_params_
    best_params['scale_pos_weight']= yields_BKGoverSIG
    best_params['eval_metric']     = "logloss"
    return best_params

def weighted_auc(y_true, y_pred, weights):
    from scipy.integrate import simps
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.array(weights)

    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)
    y_sorted = y_true[sorted_indices]
    w_sorted = weights[sorted_indices]

    # Compute cumulative true positive and false positive weights
    total_positive_weight = np.sum(w_sorted[y_sorted == 1])
    total_negative_weight = np.sum(w_sorted[y_sorted == 0])

    cum_tpr = np.cumsum(w_sorted * (y_sorted == 1)) / total_positive_weight
    cum_fpr = np.cumsum(w_sorted * (y_sorted == 0)) / total_negative_weight

    cum_fpr = cum_fpr + np.random.uniform(0, 1e-10, size=cum_fpr.shape)


    # Compute AUC using trapezoidal rule
    auc = simps(cum_fpr,cum_tpr)
    return auc

def feature_importance(model,axis, imp_type='gain',i=0, labels=None):
    # Get feature importance based on 'gain'
    feature_importance_gain = model.get_booster().get_score(importance_type=imp_type)
    # Sort features by importance (gain)
    sorted_features = sorted(feature_importance_gain.items(), key=lambda x: x[1], reverse=True)
    print(sorted_features)
    # Separate the names and importances for plotting
    sorted_feature_names = [feature for feature , importance in sorted_features]
    sorted_importances = [importance for feature , importance in sorted_features]
    if labels:
        sorted_feature_names = [labels[feature] for feature in sorted_feature_names]
    # Plot the feature importances based on gain
    #plt.figure(figsize=(10, 6))
    bars = axis.barh(sorted_feature_names, sorted_importances, color='skyblue')
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.set_title(f"Feature Importances ({imp_type}) - Fold {i}")
    axis.invert_yaxis()  # To show the highest importance at the top
    #plt.show()

    for bar in bars:
        axis.text(bar.get_width() + 0.01, 
                 bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.4f}',
                 va='center', ha='left', fontsize=14)



def load_root_files(signal_paths, background_paths, tree_name="tree_Hmumu"):
    """Load ROOT files and return a combined DataFrame."""
    def process_file(file_path, label):
        file = uproot.open(file_path)
        tree = file[tree_name]
        df = tree.arrays(FEATURES[:-1] + ["weight", "Event_MET_Phi", "Z_Phi_FSR","event"], library="pd")
        df["Process"] = (file_path.split("/")[1]).split("_")[0]
        df["label"] = label
        df["DPHI_MET_DIMU"] = np.minimum(
            2 * np.pi - abs(df["Event_MET_Phi"] - df["Z_Phi_FSR"]),
            abs(df["Event_MET_Phi"] - df["Z_Phi_FSR"])
        )
        df["fold"] = df["event"]%4
        return df

    signal_dfs     = process_file(signal_paths, label=1)# for sg_path in signal_paths]
    background_dfs = [process_file(bg_path, label=0) for bg_path in background_paths]
    return pd.concat([signal_dfs] + background_dfs, ignore_index=True)

def get_hyperparameters_table(xgb_models):
    # List of XGBoost models

    # Parameters to extract
    list_params = ["n_estimators", "learning_rate", "max_depth", "subsample", 
               "colsample_bytree", "reg_alpha", "reg_lambda"]

    # Parameter search ranges (you can modify this based on your search space)
    param_ranges = {
        "n_estimators": r"\texttt{randint(50, 500, 5)}",
        "learning_rate": r"\texttt{uniform(0.01, 0.3, 5)}",
        "max_depth": r"\texttt{randint(3, 10, 5)}",
        "subsample": r"\texttt{uniform(0.6, 1.0, 5)}",
        "colsample_bytree": r"\texttt{uniform(0.6, 1.0, 5)}",
        "reg_alpha": r"\texttt{uniform(0, 1, 5)}",
        "reg_lambda": r"\texttt{uniform(0, 1, 5)}",
    }

    # Initialize a dictionary to store table data
    table_data = {"Parameter": [], "Range": []}

    # Fill in parameter names and their corresponding ranges
    for param in list_params:
        table_data["Parameter"].append(r"\textbf{" + param.replace("_", r"\_") + r"}")
        table_data["Range"].append(param_ranges.get(param, "N/A"))

    # Extract parameter values from each model
    for i, model in enumerate(xgb_models):
        table_data[f"Model {i+1}"] = [model.get_params().get(param, "N/A") for param in list_params]

    # Convert to a Pandas DataFrame
    df = pd.DataFrame(table_data)

    # Convert DataFrame to LaTeX table
    latex_table = df.to_latex(index=False, escape=False, column_format="l l c c c c")

    # Print or save the LaTeX table
    print(latex_table)

# Load dataset
dataset_df = load_root_files(
    "inputfiles/signal_116m133.root", #,"inputfiles/signal_other_120m130.root"],
    [
        "inputfiles/TOP_116m133.root", "inputfiles/diboson_116m133.root",
        "inputfiles/dy_116m133.root"
    ]
)



# Feature processing
#FEATURES.append("DPHI_MET_DIMU")
dataset_df[FEATURES] = dataset_df[FEATURES].clip(
    lower=dataset_df[FEATURES].quantile(0.01),
    upper=dataset_df[FEATURES].quantile(0.99),
    axis=1
)




dataset_df = dataset_df.sample(frac=1,random_state=42).reset_index(drop=True)


dataset_df
print(dataset_df)


X, y, w = dataset_df[FEATURES+["fold"]], dataset_df["label"], dataset_df["weight"]

# Train-validation-test split
#X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
#    X, y, w, test_size=0.4, random_state=42, stratify=y
#)
#X_valid, X_test, y_valid, y_test, w_valid, w_test = train_test_split(
#    X_temp, y_temp, w_temp, test_size=0.5, random_state=42, stratify=y_temp
#)

# XGBoost parameters
best_parameters_RandomSearch   = {'subsample':0.96574, 'reg_lambda':0.46422, 'reg_alpha':0.24293, 'n_estimators':413, 'max_depth':6, 'learning_rate':0.22646, 'colsample_bytree':0.816356,'scale_pos_weight': yields_BKGoverSIG,'eval_metric': "logloss"}
best_parameters_RandomSearch_new= {'subsample': 0.8787698520104715, 'reg_lambda': 0.7695437229034673, 'reg_alpha': 0.3324674089489945, 'n_estimators': 469, 'max_depth': 3, 'learning_rate': 0.18876707906578757, 'colsample_bytree': 0.90998711234651,'scale_pos_weight': yields_BKGoverSIG,'eval_metric': "logloss"}
best_parameters_optuna_accuracy = {'max_depth': 3, 'learning_rate': 0.014266596240027287, 'n_estimators': 120, 'subsample': 0.6660173666440341, 'colsample_bytree': 0.9411369892349853, 'reg_alpha': 0.25816005233111994, 'reg_lambda': 0.5922340802106643, 'scale_pos_weight': yields_BKGoverSIG,'eval_metric': "logloss"}
best_parameters_optuna_TPoverFP = {'max_depth': 10, 'learning_rate': 0.01517972907975863, 'n_estimators': 127, 'subsample': 0.9229601000478028, 'colsample_bytree': 0.9277345793628416, 'reg_alpha': 0.9244762896292463, 'reg_lambda': 0.5325314327773298, 'scale_pos_weight': yields_BKGoverSIG,'eval_metric': "logloss"}
best_parameters_optuna_f1_ntr300 = {'max_depth': 4, 'learning_rate': 0.010010850482677202, 'n_estimators': 216, 'subsample': 0.8466920637169526, 'colsample_bytree': 0.910226366069031, 'reg_alpha': 0.8076938662652315, 'reg_lambda': 2.9462155956532102e-05, 'scale_pos_weight': yields_BKGoverSIG,'eval_metric': "logloss"}
best_parameters_optuna_f1_ntr300_aucpr = {'max_depth': 10, 'learning_rate': 0.01394559231685643, 'n_estimators': 105, 'subsample': 0.9996481021873942, 'colsample_bytree': 0.992631576265379, 'reg_alpha': 0.9891836161778786, 'reg_lambda': 0.9886744234520205, 'scale_pos_weight': yields_BKGoverSIG,'eval_metric': "aucpr"}


# Initialize model
#best_xgb_model = xgb.XGBClassifier(**best_parameters_RandomSearch_new, use_label_encoder=False)

# Cross-validation setup
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)


cv_results, train_losses, valid_losses = [], [], []
mean_fpr = np.linspace(0, 1, 100)
tprs_train, tprs_test, aucs_train, aucs_test = [], [], [], []
bdt_scores, event_ws, true_labels = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(y))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.tight_layout()
axes = axes.flatten()
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 15))
fig2.tight_layout()
axes2 = axes2.flatten()
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10))
fig3.tight_layout()
axes3 = axes3.flatten()
fig4, axes4 = plt.subplots(2, 2, figsize=(15, 10))
fig4.tight_layout()
axes4 = axes4.flatten()

bdt_predictions = np.zeros((X.shape[0], kf.get_n_splits()))  # Rows: events in new_data, Columns: 5 folds

all_val_indices = []

X_update=  X.copy()
#print(X_update.columns)
#best_xgb_model=[0,0,0,0,0]
# Cross-validation loop
for fold in range(4):#, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    #if fold==0: continue
    train_idx = X_update[X_update['fold'] != fold].index
    val_idx = X_update[X_update['fold'] == fold].index

    print(f"Running fold {fold}")
    print(f"Fold {fold} - Training set size: {len(train_idx)}, Validation set size: {len(val_idx)}")
    all_val_indices.extend(val_idx)

    X_train_fold, X_valid_fold = X.drop(columns=['fold']).iloc[train_idx].copy(), X.drop(columns=['fold']).iloc[val_idx].copy()
    y_train_fold, y_valid_fold = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
    w_train_fold, w_valid_fold = w.iloc[train_idx].copy(), w.iloc[val_idx].copy()
    plot_test_train_distributions(X_train_fold, y_train_fold, w_train_fold, X_valid_fold, y_valid_fold, w_valid_fold, fold=fold, title=fr"Weighted variables --Fold {fold}")

    '''
    y_train_fold = y_train_fold[w_train_fold > 0]
    X_train_fold = X_train_fold[w_train_fold > 0]
    w_train_fold = w_train_fold[w_train_fold > 0]
    y_valid_fold = y_valid_fold[w_valid_fold > 0]
    X_valid_fold = X_valid_fold[w_valid_fold > 0]
    w_valid_fold = w_valid_fold[w_valid_fold > 0]

    #plot_test_train_distributions(X_train_fold, y_train_fold, w_train_fold, X_valid_fold, y_valid_fold, w_valid_fold, f"posweights_{fold}", title=fr"Weighted variables [w>0]--Fold {fold}")
    #continue

    #print("Signal")
    #print(f"Train: {np.sum(y_train_fold==1)} , Test: {np.sum(y_valid_fold==1)}")
    #print(fold, X_valid_fold[X_valid_fold["Process"]=="dy"]["Process"].shape, X_valid_fold[X_valid_fold["Process"]=="TOP"]["Process"].shape, X_valid_fold[X_valid_fold["Process"]=="diboson"]["Process"].shape)
    #print(f"Train: {np.sum(y_train_fold==0)} , Test: {np.sum(y_valid_fold==0)}")

    #if fold==0:   parameters =  {'subsample': 0.8834580608652794, 'reg_lambda': 0.7622904742769844, 'reg_alpha': 0.4491470796888313, 'n_estimators': 405, 'max_depth': 9, 'learning_rate': 0.19090554173655913, 'colsample_bytree': 0.7799619405014517, 'scale_pos_weight': yields_BKGoverSIG, 'eval_metric': "logloss"}
    #elif fold==1: parameters =  {'subsample': 0.8787512609872534, 'reg_lambda': 0.10965882676546301, 'reg_alpha': 0.2458150419716879, 'n_estimators': 442, 'max_depth': 7, 'learning_rate': 0.05505420947526505, 'colsample_bytree': 0.8895896572119786, 'scale_pos_weight': yields_BKGoverSIG, 'eval_metric': "logloss"}
    #elif fold==2: parameters =  {'subsample': 0.9879639408647978, 'reg_lambda': 0.19967378215835974, 'reg_alpha': 0.13949386065204183, 'n_estimators': 459, 'max_depth': 7, 'learning_rate': 0.03899272558722083, 'colsample_bytree': 0.7164916560792167, 'scale_pos_weight': yields_BKGoverSIG, 'eval_metric': "logloss"}
    #else:         parameters=  {'subsample': 0.9466594563596801, 'reg_lambda': 0.26230482259638144, 'reg_alpha': 0.08250004969715541, 'n_estimators': 391, 'max_depth': 5, 'learning_rate': 0.07977098978995113, 'colsample_bytree': 0.9372898776918819, 'scale_pos_weight': yields_BKGoverSIG, 'eval_metric': "logloss"}

    parameters= get_opt_parameters(X_train_fold, y_train_fold, w_train_fold, num_iter=100, random_seed=fold)    
    best_xgb_model[fold] = xgb.XGBClassifier(**parameters, use_label_encoder=False)

    best_xgb_model[fold].fit(
        X_train_fold, y_train_fold, sample_weight=w_train_fold,
        eval_set=[(X_train_fold, y_train_fold), (X_valid_fold, y_valid_fold)],
        verbose=False
    )
    best_xgb_model[fold].save_model(f'xgboost_bestmodel_fold{fold}.json')
    continue
    '''
    #best_xgb_model[fold] = xgb.XGBClassifier()
    #best_xgb_model[fold].load_model(f'xgboost_bestmodel_fold{fold}.json')
    evals_result = best_xgb_model[fold].evals_result()
    train_losses.append(evals_result['validation_0']['logloss'][-1])
    valid_losses.append(evals_result['validation_1']['logloss'][-1])

    y_train_pred_prob = best_xgb_model[fold].predict_proba(X_train_fold)[:, 1]
    y_valid_pred_prob = best_xgb_model[fold].predict_proba(X_valid_fold)[:, 1]

    # Now assign the BDT scores to the corresponding rows in dataset_df for this fold
    bdt_predictions[:, fold ] = best_xgb_model[fold].predict_proba(X.drop(columns=["fold"]))[:, 1]  # Store in the corresponding column
    X_update.loc[val_idx, 'BDT_score'] = best_xgb_model[fold].predict_proba(X_valid_fold)[:, 1]

    fpr_train, tpr_train, _ = roc_curve(y_train_fold, y_train_pred_prob, sample_weight=w_train_fold)
    fpr_valid, tpr_valid, _ = roc_curve(y_valid_fold, y_valid_pred_prob, sample_weight=w_valid_fold)

    tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
    tprs_test.append(np.interp(mean_fpr, fpr_valid, tpr_valid))
    aucs_train.append(weighted_auc(y_train_fold, y_train_pred_prob,w_train_fold))
    aucs_test.append(weighted_auc(y_valid_fold, y_valid_pred_prob, w_valid_fold))

    #cv_results.append(accuracy_score(y_valid_fold, best_xgb_model[fold].predict(X_valid_fold)))  This has unweighted accuracy

    # Plot loss curve
    ax = axes[fold]
    #ax.plot(evals_result['validation_0']['logloss'], label="Train", color="blue")
    #ax.plot(evals_result['validation_1']['logloss'], label="Test", color="red")
    ax.set_title(f"Loss - Fold {fold}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log Loss")
    ax.legend()
    ax.grid()
    # Plot confusion matrix   
    cm = confusion_matrix(y_valid_fold, best_xgb_model[fold].predict(X_valid_fold),sample_weight=w_valid_fold)
    cv_results.append( (cm[1,1]+cm[0,0])/(cm[1,1]+cm[1,0]+cm[0,1]+cm[0,0]) )
    precision =  cm[1,1]/ (cm[1,1]+cm[0,1])
    recall    =  cm[1,1]/ (cm[1,1]+cm[1,0])
    f1score   = 2*precision*recall/(precision+recall)
    ax = axes2[fold]
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Background", "Signal"], yticklabels=["Background", "Signal"],ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Test Fold {fold})")
    metrics_text = (
    f"Metrics Summary\n-----------------\n"
    f"Precision: {precision:.4f}\n" 
    f"Recall: {recall:.3f}\n"
    f"F1-score: {f1score:.4f}\n"
    f"Accuracy: {100*cv_results[fold]:.1f}%"
    )
    # Place the metrics below the heatmap
    ax.text(0.85, -0.20, metrics_text, ha='center', va='top', fontsize=14, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Plot ROC Curves
    ax= axes3[fold]
    ax.plot(fpr_train, tpr_train, label=f"Train (AUC={aucs_train[fold]:.3f}", color="blue")
    ax.plot(fpr_valid, tpr_valid, label=f"Test (AUC={aucs_test[fold]:.3f})", color="red")
    ax.plot([0,1],[0,1], label="AUC=0.5", color='k', ls='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(f"ROC Curve (Fold {fold})")
    ax.legend()
    ax.grid
    #plot feature importance
    ax= axes4[fold]
    feature_importance(best_xgb_model[fold],ax,i=fold,labels=labels)
print(f"Total validation indices across all folds: {len(set(all_val_indices))} (should be same as total number of validation events)")
# Show plot
fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()

fig.savefig("vh2lep_BDT_lossfunc.pdf", format="pdf")
fig2.savefig("vh2lep_BDT_confusionmatrix.pdf", format="pdf")
fig3.savefig("vh2lep_BDT_ROCcurves.pdf", format="pdf")
fig4.savefig("vh2lep_BDT_FeatureImportance.pdf", format="pdf")

plt.show()
print(best_xgb_model[:-1])
get_hyperparameters_table(best_xgb_model[:-1])

dataset_df["BDT_avg"] = np.mean(bdt_predictions, axis=1)
dataset_df_new =pd.concat([X_update, dataset_df["BDT_avg"],dataset_df["Process"], w,y], axis=1)
#X_update[y == 1]
Signal= dataset_df_new[dataset_df_new["label"]==1]
Backgr= dataset_df_new[dataset_df_new["label"]==0]
DY    = Backgr[Backgr["Process"]=="dy"]
VV    = Backgr[Backgr["Process"]=="diboson"]
top   = Backgr[Backgr["Process"]=="TOP"]

Top = pd.concat([top,VV])

dataset_df_other_signal = load_root_files("inputfiles/signal_other_116m133.root",[])
dataset_df_other_signal[FEATURES]
dataset_df_other_signal = dataset_df_other_signal.sample(frac=1,random_state=45).reset_index(drop=True)

Xo,yo,wo= dataset_df_other_signal[FEATURES], dataset_df_other_signal["label"], dataset_df_other_signal["weight"]
dataset_df_other_signal["fold"] = dataset_df_other_signal.index % 4  # This assigns a fold index (0-3)

# Step 2: Initialize the BDT_score column
dataset_df_other_signal["BDT_score"] = np.nan  

# Step 3: Apply the trained XGB models to each fold
for i in range(4):
    mask = dataset_df_other_signal["fold"] == i  # Select the subset for this fold
    dataset_df_other_signal.loc[mask, "BDT_score"] = best_xgb_model[i].predict_proba(dataset_df_other_signal.loc[mask, FEATURES])[:,1]

# Step 4: Remove the temporary "fold" column
dataset_df_other_signal = dataset_df_other_signal.drop(columns=["fold"])
rgb_vv= (102/255., 255/255., 102/255.)
rgb_top=(108/255., 187/255., 108/255.)
rgb_dy=(0, 255/255., 0)

Signal["BDT_score"].hist(bins=200, color='black', linewidth=2, histtype='step' ,weights=Signal["weight"], label='Signal')
Backgr["BDT_score"].hist(bins=200, color=rgb_dy,weights=Backgr["weight"], label=r"$Z\to\mu\mu$")
Top["BDT_score"].hist(bins=200, color=rgb_top,weights=Top["weight"], label="Top")
VV["BDT_score"].hist(bins=200, color=rgb_vv,weights=VV["weight"], label="Diboson")
plt.hist(Backgr["BDT_score"], bins=200, weights=Backgr["weight"], histtype='step', color='black', linewidth=1)
plt.hist(Top["BDT_score"], bins=200, weights=Top["weight"], histtype='step', color='black', linewidth=1)
plt.hist(VV["BDT_score"], bins=200, weights=VV["weight"], histtype='step', color='black', linewidth=1)

dataset_df_other_signal["BDT_score"].hist(bins=200, color='r', linewidth=2, histtype='step' ,weights=dataset_df_other_signal["weight"], label='H (Other)')
plt.rcParams.update({"font.size": 14}) 

plt.xlabel("BDT Score")
plt.ylabel("Events")
plt.yscale('log')
plt.legend()
plt.title("BDT Score Distribution")
plt.ylim(0.0003,70)
plt.savefig("vh2lep_BDT_distribution.pdf", format="pdf")
plt.show()
print(cv_results)




# In[98]:


# Custom dictionary for node labels (key: node ID, value: custom label)
custom_labels = {
    # Add more mappings for your specific tree
    0: r"$\Delta\Phi_{\nu\nu,\mu\mu}<3.0872068$",
    1: r"$\Delta\Phi_{\nu\nu,\mu\mu}<0.138401151$"
}

# Extract the tree dump (all trees in the model)
dump = best_xgb_model[2].get_booster().get_dump()[-1]
print(dump)
# Define a function to update the tree dump with custom labels
def update_tree_with_labels(tree_dump, custom_labels):
    updated_dump = []
    for tree in tree_dump:
        updated_tree = tree
        for node_id, label in custom_labels.items():
            updated_tree = updated_tree.replace(f"f{node_id}", label)  # Replace default labels with custom labels
        updated_dump.append(updated_tree)
    return updated_dump

# Apply the custom labels to the tree dump
updated_dump = update_tree_with_labels(dump, custom_labels)

params = {
    'condition_node_params' : {'shape': 'box', 'style': 'filled,rounded','fillcolor': '#78bceb'},
    'leaf_node_params'      : {'shape': 'box', 'style': 'filled', 'fillcolor': '#e48038'},
    'yes_color' : 'red',
    'no_color'  : 'green'
}

xgb.plot_tree(best_xgb_model[2],filled=True, num_trees=len(best_xgb_model[2].get_booster().get_dump())-1, **params )
xgb.plot_tree(best_xgb_model[2],filled=True, num_trees=0, **params )

plt.show()



# BDT Scores
plt.figure(figsize=(8, 5))
#plt.hist(dataset_df[true_labels == 1], weights=dataset_df[true_labels == 1], bins=50, alpha=0.5, color="blue", label="Signal")
#plt.hist(bdt_scores[true_labels == 0], weights=event_ws[true_labels == 0], bins=50, alpha=0.5, color="red", label="Background")
Signal = dataset_df[dataset_df["label"] == 1]
Backgr = dataset_df[dataset_df["label"] == 0]

Signal["BDT_score"].hist(bins=200, color='r', alpha=0.25,weights=Signal["weight"], label='Signal')
Backgr["BDT_score"].hist(bins=200, color='b', alpha=0.25,weights=Backgr["weight"], label='Background')
plt.xlabel("BDT Score")
plt.ylabel("Events")
plt.yscale('log')
plt.legend()
plt.title("BDT Score Distribution")
plt.show()


# In[156]:


bins=200
# Compute weighted histograms
sig_hist, bin_edges = np.histogram(Signal["BDT_score"], bins=bins, weights=Signal["weight"])
bkg_hist, _ = np.histogram(Backgr["BDT_score"], bins=bins, weights=Backgr["weight"])

# Compute cumulative histograms
sig_cum = np.cumsum(sig_hist[::-1])[::-1]
bkg_cum = np.cumsum(bkg_hist[::-1])[::-1]

# Compute ratio (avoid division by zero)
ratio = np.divide(sig_cum, np.sqrt(bkg_cum), out=np.zeros_like(sig_cum), where=bkg_cum > 0)

# Plot cumulative histograms
plt.figure(figsize=(10, 6))
plt.plot(bin_edges[:-1], sig_cum, label="Cumulative Signal", color="r")
plt.plot(bin_edges[:-1], bkg_cum, label="Cumulative Background", color="b")
plt.xlabel("BDT Score")
plt.ylabel("Cumulative Events")
plt.yscale("log")
plt.legend()
plt.title("Cumulative BDT Score Distribution")
plt.show()

# Plot ratio: Cumulative Signal / sqrt(Cumulative Background)
plt.figure(figsize=(10, 6))
plt.plot(bin_edges[:-1], ratio, color="k", label=r"$S / \sqrt{B}$")
plt.xlabel("BDT Score")
plt.xlim(0.8,1.0)
plt.ylabel("Ratio $S / \sqrt{B}$")
plt.title("Significance vs. BDT Score")
plt.axhline(0.1, color="gray", linestyle="dashed")  # Reference line
plt.legend()
plt.show()


# In[60]:


print("Initial")
#sig_other = 
Stot  = np.sum(Signal[Signal["BDT_score"]>=0]["weight"])
Btot  = np.sum(Backgr[Backgr["BDT_score"]>=0]["weight"])
Ztot   = Stot/np.sqrt(Btot)
print("S = ",Stot)
print("B = ",Btot)
print("Z = ",Ztot )
print("============")
print("Cut-Based selection")
Scut= np.sum( Signal[(Signal["Event_VT_over_HT"]>0.5)&(Signal["DPHI_MET_DIMU"]>2.3)&(Signal["dR_mu0_mu1"]>0.5)&(Signal["dR_mu0_mu1"]<1.75)&(Signal["Event_MET"]>120)]["weight"] )
Bcut= np.sum( Backgr[(Backgr["Event_VT_over_HT"]>0.5)&(Backgr["DPHI_MET_DIMU"]>2.3)&(Backgr["dR_mu0_mu1"]>0.5)&(Backgr["dR_mu0_mu1"]<1.75)&(Backgr["Event_MET"]>120)]["weight"] )
Zcut= Scut/np.sqrt(Bcut)
print("S = ",Scut )
print("B = ",Bcut )
print("Z = ",Zcut  )
print("============")
thresholds=np.linspace(0,1,201)
signif = []
signifAs=[]
sig_eff =[]
bkg_rej =[]
Sbest, S5=0,0
Bbest, S5=0,0
max_sig =0
for threshold in thresholds:
    #print(f"BDT Selection (>{threshold})")
    S = np.sum(Signal[Signal["BDT_score"]>threshold]["weight"])
    S_others_sum = S+ np.sum(dataset_df_other_signal[dataset_df_other_signal["BDT_score"]>threshold]["weight"])
    B = np.sum(Backgr[Backgr["BDT_score"]>threshold]["weight"])
    if B>=5: background_threshold , S5, B5= threshold, S, B 
    Z = S / np.sqrt(B) if B!=0 else 0
    Z2= np.sqrt( 2 *( (S+B)*np.log(1+ (S/B)) - S )) if B!=0 else 0
    #print(f"Z = {Z:.5f} (BDT>{threshold:.3f} --> S={S:.3f} B={B:.3f})" )
    signif.append(Z)
    signifAs.append(Z2)
    sig_eff.append(S/Stot)
    bkg_rej.append(Btot/B if B!=0 else 0)
    max_sig = max(max_sig, Z)
    if round(Z,4)==round(max_sig,4): 
        Sbest,Sbest_total, Bbest,threshold_best = S, S_others_sum, B, threshold
print(f"Upper threshold (Best): {threshold_best:.3f}\n S={Sbest:.3f}  B={Bbest:.3f}  Z={Sbest/np.sqrt(Bbest):.5f} ")
print(f"Upper threshold (Best) + Other H signals:\n S={Sbest_total:.3f}  B={Bbest:.3f}  Z={Sbest_total/np.sqrt(Bbest):.5f} ")

print(threshold_best,Sbest/Bbest**0.5)
plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 14}) 
plt.plot(thresholds, signifAs, color="k", label=r"Asimov Significance")
plt.plot(thresholds, signif , color="r", ls='--', label=r"$S / \sqrt{B}$")
#plt.vlines(background_threshold, 0, 0.1, label="Threshold B<5")
plt.hlines(max_sig, 0,threshold_best, label="Max significance", color='k',ls=':')
plt.vlines(threshold_best, 0,max_sig, color='k',ls=':')

plt.text(0.805, 0.07, f'Z[max] = {max_sig:.4f}',fontsize=15, color='k')
plt.xlabel("BDT Score Threshold")
plt.xlim(0.8,1.0)
plt.ylim(0.0,0.1)
plt.ylabel(r"Significance, $Z$")
plt.title("Significance vs. BDT Score Threshold")
#plt.axhline(0.1, color="gray", linestyle="dashed")  # Reference line
plt.legend()
plt.grid()
plt.savefig("vh2lep_BDT_significancesvsbdtup.pdf", format="pdf")
plt.show()

X = np.linspace(0,1,200)
target_Z = 0.09
Z_sig_line= (target_Z/(Ztot*X))**2
Z_sig_line0= ((Scut/np.sqrt(Bcut))/(Ztot*X))**2
Z_sig_line1= (max_sig/(Ztot*X))**2
Z_sig_line2= (0.090/(Ztot*X))**2
Z_sig_line3= (0.15 /(Ztot*X))**2

plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 14}) 

plt.plot(sig_eff, bkg_rej, color="b", label=r"BDT selection")
plt.plot(X, Z_sig_line0, ls='--',color="r", label=rf"Z={Scut/Bcut**0.5:.3f} (Cut)")
plt.plot(X, Z_sig_line1, ls='--',color="b", label=rf"Z={Sbest/Bbest**0.5:.3f} (2lep)")
plt.plot(X, Z_sig_line2, ls='--',color="k", label=r"Z=0.090 (4lep)")
plt.plot(X, Z_sig_line3, ls='--',color="g", label=r"Z=0.150 (3lep)")

plt.plot([Scut/Stot], [Btot/Bcut],  marker='o',ms = 10, mec = 'r', color='white',mfc = 'r', label='Cut-based selection')
plt.plot([Sbest/Stot], [Btot/Bbest],  marker='^',ms = 10, mec = 'b', color='white',mfc = 'b', label='BDT highest Z')
#plt.plot([S5/Stot], [Btot/B5],  marker='<',ms = 10, mec = 'g', color='white',mfc = 'g', label=r'BDT $N_{B} \approx 5$')

plt.xlabel("Signal Efficiency")
plt.xlim(0.0,1.0)
plt.ylim(1.0,10000.0)
plt.yscale('log')
plt.ylabel("Background Rejection")
plt.title("Significance vs. BDT Score Threshold")
#plt.axhline(0.1, color="gray", linestyle="dashed")  # Reference line
plt.legend()
plt.grid()
plt.savefig("vh2lep_BDT_sigeffvsbkgrej.pdf", format="pdf")
plt.show()



# In[57]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np
model = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.01,use_label_encoder=False, eval_metric='logloss', scale_pos_weight=yields_BKGoverSIG )
# Define parameter grid with distributions
param_dist = {
    'max_depth': np.random.randint(3, 10, 5),
    'learning_rate': np.random.uniform(0.01, 0.3, 5),
    'n_estimators': np.random.randint(100, 500, 5),
    'subsample': np.random.uniform(0.6, 1.0, 5),
    'colsample_bytree': np.random.uniform(0.6, 1.0, 5),
    'reg_alpha': np.random.uniform(0, 1, 5),
    'reg_lambda': np.random.uniform(0, 1, 5)
}

# Perform randomized search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=40, scoring='balanced_accuracy', cv=5, verbose=1, random_state=42)
random_search.fit(X_train[w_train>0], y_train[w_train>0],sample_weight=w_train[w_train>0])

# Best parameters
print("Best parameters:", random_search.best_params_)
best_parameters_RFo = random_search.best_params_


# In[187]:


import itertools


plt.rcParams.update({"font.size": 14}) 
# Separate signal and background datasets
signal_df = dataset_df[dataset_df["label"] == 1]
bkg_df = dataset_df[dataset_df["label"] == 0]

# Define number of features
#FEATURES += ["Muons_CosThetaStar","Jets_PT_jj","Event_MET_Sig"]
num_features = len(FEATURES)
labels={"Muons_PT_Lead":r"$p_{T}^{\mu_1}$ [GeV]", "Muons_PT_Sub":r"$p_{T}^{\mu_2}$ [GeV]", "Event_VT_over_HT":r"$V_{T}/H_{T}$", "DPHI_MET_DIMU":r"$\Delta\phi (\mu\mu, E_{T}^{miss})$ [rad]" ,
        "dR_mu0_mu1":r"$\Delta R(\mu_1, \mu_2)$", "Jets_jetMultip":r"$N_{j}$", "Event_MET":r"$E_{T}^{miss}$ [GeV]","Muons_CosThetaStar":r"$cos \theta^{*}$","Event_MET_Sig":r"$\sigma (E_{T}^{miss}$)"}

# 1️⃣ **Feature Distributions**
plt.figure(figsize=(15, num_features * 5))  # Adjust figure size based on number of features
for i, feature in enumerate(FEATURES):
    plt.subplot(num_features, 2, i + 1)
    sns.histplot(signal_df[feature],  bins=50, color="r", label="Signal", kde=True, stat="density", alpha=0.5)
    sns.histplot(bkg_df[feature], bins=50, color="b", label="Background", kde=True, stat="density", alpha=0.5)
    plt.xlabel(labels[feature])
    plt.ylabel("Density")
    plt.legend()
plt.tight_layout()
plt.savefig("vh2lep_BDT_features.pdf", format="pdf")
plt.show()

# 2️⃣ **Scatter Plots for Feature Pairs**
scatter_fig, scatter_axes = plt.subplots(num_features, num_features, figsize=(12, 12), sharex="col", sharey="row")

for i, feature_i in enumerate(FEATURES):
    for j, feature_j in enumerate(FEATURES):
        ax = scatter_axes[i, j]
        if i == j:
            # Diagonal: Draw a histogram of the feature instead of scatter
            sns.histplot(signal_df[feature_i], bins=50, color="r", kde=True, stat="density", alpha=0.5, ax=ax)
            sns.histplot(bkg_df[feature_i], bins=50, color="b", kde=True, stat="density", alpha=0.5, ax=ax)
        else:
            ax.scatter(bkg_df[feature_j], bkg_df[feature_i], marker='.', s=0.001, color='b', alpha=0.4, label="Background")
            ax.scatter(signal_df[feature_j], signal_df[feature_i], marker='.', s=0.001, color='r', label="Signal")

        if j == 0:
            ax.set_ylabel(labels[feature_i])
        if i == num_features - 1:
            ax.set_xlabel(labels[feature_j])

plt.tight_layout()
plt.show()


# In[162]:


signif = []
BDT_low = 0.800
BDT_high= 0.960
thresholds=np.linspace(0,BDT_high,1+int(BDT_high/0.005))
signal_leftover = Signal[Signal["BDT_score"]<0.955]
backgr_leftover = Backgr[Backgr["BDT_score"]<0.955]
for threshold in thresholds:
    S = np.sum(signal_leftover[signal_leftover["BDT_score"]>threshold  ]["weight"])
    B = np.sum(backgr_leftover[backgr_leftover["BDT_score"]>threshold  ]["weight"])
    Z = S / np.sqrt(B) if B!=0 else 0
    #print(f"Z = {Z:.5f} (BDT>{threshold:.3f} --> S={S:.3f} B={B:.3f})" )
    signif.append(Z)
#plt.plot(thresholds, signif)
#plt.show()




Signal["BDT_score"].hist(bins=200, color='r', alpha=0.25,weights=w[y==1], label='Signal')
Backgr["BDT_score"].hist(bins=200, color='b', alpha=0.25,weights=w[y==0], label="Background")
plt.plot([ BDT_low , BDT_low], [0, 100], ls='--', color='green')
plt.plot([ BDT_high , BDT_high], [0, 100], ls='--', color='cyan')
plt.xlabel("BDT Score")
plt.ylabel("Events")
plt.yscale('log')
plt.legend()
plt.title("BDT Score Distribution")
plt.show()


S_High = np.sum(Signal[Signal["BDT_score"]>BDT_high  ]["weight"])
S_Low = np.sum(signal_leftover[signal_leftover["BDT_score"]>BDT_low  ]["weight"])
S_Rest = np.sum(signal_leftover[signal_leftover["BDT_score"]<BDT_low  ]["weight"])

B_High = np.sum(Backgr[Backgr["BDT_score"]>BDT_high  ]["weight"])
B_Low  = np.sum(backgr_leftover[backgr_leftover["BDT_score"]>BDT_low  ]["weight"])
B_Rest = np.sum(backgr_leftover[backgr_leftover["BDT_score"]<BDT_low  ]["weight"])

print(f"======= Significances =========\nBDT[{BDT_high:.3f},1.000] >>> S_High = {S_high:.3f}, B_High = {B_high:.3f}, Z = {S_High/B_High**0.5:.5f}\n ")
print(f"BDT[{BDT_low:.3f},{BDT_high:.3f}] >>> S_Low  = {S_Low:.3f} , B_Low  = {B_low:.3f} , Z = {S_Low/B_Low**0.5:.5f}  \n")
print(f"Total = { np.sqrt( (S_Low**2)/B_Low + (S_High**2)/B_High ) :.5f}")
print(f"BDT[0.000,{BDT_low:.3f}] >>> S_Rest = {S_Rest:.3f}, B_Rest = {B_Rest:.3f}, Z = {S_Rest/B_Rest**0.5:.5f}  \n")



'''

signifAs=[]
sig_eff =[]
bkg_rej =[]
Sbest, S5=0,0
Bbest, S5=0,0
max_sig =0
for threshold in thresholds:
    #print(f"BDT Selection (>{threshold})")
    S = np.sum(Signal[Signal["BDT_score"]>threshold]["weight"])
    B = np.sum(Backgr[Backgr["BDT_score"]>threshold]["weight"])
    if B>=5: background_threshold , S5, B5= threshold, S, B 
    Z = S / np.sqrt(B) if B!=0 else 0
    Z2= np.sqrt( 2 *( (S+B)*np.log(1+ (S/B)) - S )) if B!=0 else 0
    print(f"Z = {Z:.5f} (BDT>{threshold:.3f} --> S={S:.3f} B={B:.3f})" )
    signif.append(Z)
    signifAs.append(Z2)
    sig_eff.append(S/Stot)
    bkg_rej.append(Btot/B if B!=0 else 0)
    max_sig = max(max_sig, Z)
    if round(Z,4)==round(max_sig,4): Sbest, Bbest = S,B

plt.figure(figsize=(10, 6))
plt.plot(thresholds, signifAs, color="k", label=r"Asimov Significance")
plt.plot(thresholds, signif , color="r", ls='--', label=r"$S / \sqrt{B}$")
plt.vlines(background_threshold, 0, 0.1, label="Threshold B<5")
plt.hlines(max_sig, 0,1, label="Max significance", color='k',ls=':')
plt.text(0.805, 0.07, f'Z[max] = {max_sig:.4f}',fontsize=15, color='k')
plt.xlabel("BDT Score Threshold")
plt.xlim(0.8,1.0)
plt.ylim(0.0,0.1)
plt.ylabel("Ratio $S / \sqrt{B}$")
plt.title("Significance vs. BDT Score Threshold")
#plt.axhline(0.1, color="gray", linestyle="dashed")  # Reference line
plt.legend()
plt.grid()
plt.show()

X = np.linspace(0,1,200)
target_Z = 0.09
Z_sig_line= (target_Z/(Ztot*X))**2
Z_sig_line0= ((Scut/np.sqrt(Bcut))/(Ztot*X))**2
Z_sig_line1= (0.067/(Ztot*X))**2
Z_sig_line2= (0.090/(Ztot*X))**2
Z_sig_line3= (0.15 /(Ztot*X))**2

plt.figure(figsize=(10, 6))
plt.plot(sig_eff, bkg_rej, color="b", label=r"BDT selection")
plt.plot(X, Z_sig_line0, ls='--',color="r", label=rf"Z={Scut/Bcut**0.5:.3f} (Cut)")
plt.plot(X, Z_sig_line1, ls='--',color="b", label=rf"Z={Sbest/Bbest**0.5:.3f} (2lep)")
plt.plot(X, Z_sig_line2, ls='--',color="k", label=r"Z=0.090 (4lep)")
plt.plot(X, Z_sig_line3, ls='--',color="g", label=r"Z=0.150 (3lep)")

plt.plot([Scut/Stot], [Btot/Bcut],  marker='o',ms = 10, mec = 'r', color='white',mfc = 'r', label='Cut-based selection')
plt.plot([Sbest/Stot], [Btot/Bbest],  marker='^',ms = 10, mec = 'b', color='white',mfc = 'b', label='BDT highest Z')
plt.plot([S5/Stot], [Btot/B5],  marker='<',ms = 10, mec = 'g', color='white',mfc = 'g', label=r'BDT $N_{B} \approx 5$')

plt.xlabel("Signal Efficiency")
plt.xlim(0.0,1.0)
plt.ylim(1.0,10000.0)
plt.yscale('log')
plt.ylabel("Background Rejection")
plt.title("Significance vs. BDT Score Threshold")
#plt.axhline(0.1, color="gray", linestyle="dashed")  # Reference line
plt.legend()
plt.grid()
plt.show()
'''


# In[59]:


print("Initial")

'''
Stot  = np.sum(Signal[Signal["BDT_score"]>=0]["weight"])
Btot  = np.sum(Backgr[Backgr["BDT_score"]>=0]["weight"])
Ztot   = Stot/np.sqrt(Btot)
signif = []
signifAs=[]
sig_eff =[]
bkg_rej =[]
Sbest, Sup, Sdn = 0 , 0, 0 
Bbest, Bup, Bdn = 0 , 0 , 0
Sobest,Soup,Sodn = 0, 0 , 0

max_sig =0
thresholds_up=np.linspace(0.005,1,200)
for i,threshold_up in enumerate(thresholds_up):
    print(f"{i}/200")
    thresholds_low = np.linspace(0, threshold_up, int(threshold_up/0.005)+1 )
    signif.append([])
    signifAs.append([])
    sig_eff.append([])
    bkg_rej.append([])
    for j,threshold_low in enumerate(thresholds_low): 
        #Calculate upper threshold values
        Su = np.sum(Signal[Signal["BDT_score"]>=threshold_up]["weight"])
        Su_others = Su+ np.sum(dataset_df_other_signal[dataset_df_other_signal["BDT_score"]>=threshold_up]["weight"])
        Bu = np.sum(Backgr[Backgr["BDT_score"]>=threshold_up]["weight"])
        Zu = Su / np.sqrt(Bu) if Bu!=0 else 0
        Z2u= np.sqrt( 2 *( (Su+Bu)*np.log(1+ (Su/Bu)) - Su )) if Bu!=0 else 0

        #Calculate lower threshold values
        Sd = np.sum(Signal[(Signal["BDT_score"] >= threshold_low) & (Signal["BDT_score"] < threshold_up)]["weight"])
        Sd_others= Sd + np.sum(dataset_df_other_signal[(dataset_df_other_signal["BDT_score"] >= threshold_low) & (dataset_df_other_signal["BDT_score"] < threshold_up)]["weight"])
        Bd = np.sum(Backgr[(Backgr["BDT_score"]>=threshold_low) & (Backgr["BDT_score"]<threshold_up)]["weight"])
        Zd = Sd / np.sqrt(Bd) if Bd!=0 else 0
        Z2d= np.sqrt( 2 *( (Sd+Bd)*np.log(1+ (Sd/Bd)) - Sd )) if Bd!=0 else 0

        #Combine significances
        Z = np.sqrt(Zu**2 + Zd**2)
        Z2 = np.sqrt(Z2u**2 + Z2d**2)

        #print(f"Z = {Z:.5f} (BDT>{threshold:.3f} --> S={S:.3f} B={B:.3f})" )
        signif[i].append(Z)
        signifAs[i].append(Z2)
        sig_eff.append(S/Stot)
        bkg_rej.append(Btot/B if B!=0 else 0)
        max_sig = max(max_sig, Z)
        if round(Z,4)==round(max_sig,4): 
            Sup = Su
            Sdn = Sd
            Sbest = Su + Sd
            Bup = Bu
            Bdn = Bd
            Bbest = Bu + Bd
            thresholds_best = (threshold_low, threshold_up)
            Zbest = Z
            Soup = Su_others
            Sodn = Sd_others
            Sobest= Su_others+Sd_others
print(f"Upper threshold (Best): [{thresholds_best[0]:.3f},{thresholds_best[1]:.3f}] \n S={Sbest:.3f}  B={Bbest:.3f}  Z={Sbest/np.sqrt(Bbest):.5f} ")
print(f"Upper threshold (Best) + Other H signals:\n S={Sbest_total:.3f}  B={Bbest:.3f}  Z={Sbest_total/np.sqrt(Bbest):.5f} ")

# Convert to numpy arrays for heatmap
max_length = max(len(row) for row in signif)
# Pad each row to the max length
signif_padded = np.array([np.pad(row, (0, max_length - len(row)), constant_values=0) for row in signif])
'''
# Plot heatmap
plt.figure(figsize=(10, 8))
ax=sns.heatmap(signif_padded, xticklabels=False, yticklabels=False, cmap="coolwarm", cbar=True)
plt.xlabel("Lower BDT threshold")
plt.ylabel("Upper BDT threshold")
plt.title("Significance")
ax.invert_yaxis()
# Set custom tick labels
num_ticks = 5  # Number of ticks you want to display
tick_positions = np.linspace(0, len(thresholds_up) - 1, num_ticks, dtype=int)  # Select positions

ax.set_xticks(tick_positions)
ax.set_xticklabels([f"{thresholds_low[i]:.2f}" for i in tick_positions])  # Format to 2 decimals

ax.set_yticks(tick_positions)
ax.set_yticklabels([f"{thresholds_up[i]:.2f}" for i in tick_positions])
plt.vlines(150,0,1, color='k')

cbar = ax.collections[0].colorbar  # Get the color bar
cbar.set_label("Significance", fontsize=14)

metrics_text = (
    f"Summary\n-----------------\n"
    f"BDT_low : {thresholds_best[0]:.3f}\n" 
    f"BDT_high: {thresholds_best[1]:.3f}\n"
    f"Z_low : {Sdn/np.sqrt(Bdn):.4f}\n"
    f"Z_high: {Sup/np.sqrt(Bup):.4f}\n"
    f"Z_tot : {np.sqrt((Sdn**2/Bdn)+(Sup**2/Bup)):.4f}"
    )
    # Place the metrics below the heatmap
ax.text(0.75, 0.35, metrics_text, ha='center', va='top', fontsize=14, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.savefig("vh2lep_BDT_significancevsbdtthresholdsscan.pdf", format="pdf")

plt.show()

print(np.max(signif_padded))
print(f"Best thresholds (Best): [{thresholds_best[0]:.3f},{thresholds_best[1]:.3f}] \n S={Sbest:.3f}  B={Bbest:.3f}  Z={Sbest/np.sqrt(Bbest):.5f} ")
print(f"TIGHT           : S={Sup:.3f}  B={Bup:.3f}  Z={Sup/np.sqrt(Bup):.5f}  ///   Incl. other signals: S={Soup:.3f} Z={Soup/np.sqrt(Bup):.5f}")
print(f"LOOSE_not_TIGHT : S={Sdn:.3f}  B={Bdn:.3f}  Z={Sdn/np.sqrt(Bdn):.5f}  ///   Incl. other signals: S={Sodn:.3f} Z={Sodn/np.sqrt(Bdn):.5f}")

#print(f"Best thresholds (Best): [{thresholds_best[0]:.3f},{thresholds_best[1]:.3f}] \n S={Sbest:.3f}  B={Bbest:.3f}  Z={Sbest/np.sqrt(Bbest):.5f} ")

#print(f"Upper threshold (Best) + Other H signals:\n S={Sbest_total:.3f}  B={Bbest:.3f}  Z={Sbest_total/np.sqrt(Bbest):.5f} ")


# In[ ]:




