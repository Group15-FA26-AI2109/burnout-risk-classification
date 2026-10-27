"""Build the revised manuscript, figures and standalone dashboard from audited CSVs."""
from pathlib import Path
import json, re, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision_analysis import ROOT, OUT, RES, FIG, LABELS

PAPER=ROOT/'paper_revised'
PF=PAPER/'figures'
PF.mkdir(parents=True,exist_ok=True)

def tex(s):
    return str(s).replace('\\','\\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')

def rows(frame, columns, formats=None):
    formats=formats or {}
    return '\n'.join(' & '.join(formats.get(k,lambda x:tex(x))(row[k]) for k in columns)+r' \\' for _,row in frame.iterrows())

def figure(file,caption,label,width='.95'):
    return rf'\begin{{figure}}[!htbp]\centering\includegraphics[width={width}\textwidth]{{figures/{file}}}\caption{{{caption}}}\label{{{label}}}\end{{figure}}'+'\n'

def table(caption,label,columns,header,body):
    return rf'\begin{{table}}[!htbp]\caption{{{caption}}}\label{{{label}}}\centering\small\begin{{tabular}}{{{columns}}}\toprule {header} \\ \midrule'+'\n'+body+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n'

def compact_figures(models, abl, sh):
    """Draw four paper figures at their intended print size, without reducing body type."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    short={'work_hours_per_week':'Weekly hours','stress_level':'Stress',
           'work_life_balance':'Work-life balance','manager_support':'Manager support',
           'has_therapy':'Therapy','anxiety_score':'Anxiety','overtime_hours':'Overtime',
           'depression_score':'Depression'}
    with plt.rc_context({'font.size':8.5,'axes.titlesize':9,'axes.labelsize':8,
                         'xtick.labelsize':7.5,'ytick.labelsize':8}):
        fig,ax=plt.subplots(figsize=(4.8,2.35));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        positions=[(.25,.84),(.75,.84),(.75,.50),(.25,.50),(.25,.16),(.75,.16)]
        blocks=['1  Data and target\n150,000 synthetic records\nExclude burnout score/level',
                '2  Stratified split\n120,000 train / 30,000 test\nSeed 42',
                '3  Training-fold pipeline\nEncode, scale; optional SMOTE\nFit within training folds',
                '4  Training model selection\nFive classifiers; five-fold CV\nHighest mean AP',
                '5  Held-out analyses\nMetrics, paired bootstrap\nSHAP and feature ablations',
                '6  Dashboard (HTML/PDF)\nVerified SQL and CSV outputs\nDescriptive filters']
        for (x,y),text in zip(positions,blocks):
            ax.text(x,y,text,ha='center',va='center',fontsize=8,
                    bbox={'boxstyle':'round,pad=.35','fc':'#f3f6f8','ec':'#617283'})
        arrows=[((.46,.84),(.54,.84)),((.75,.70),(.75,.64)),
                ((.54,.50),(.46,.50)),((.25,.36),(.25,.30)),((.46,.16),(.54,.16))]
        for a,b in arrows:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#465d70'})
        fig.tight_layout(pad=.3);fig.savefig(FIG/'pipeline_compact.png',dpi=300,bbox_inches='tight');plt.close(fig)

        features=sh.head(8).index[::-1]
        perm=pd.read_csv(RES/'permutation_importance.csv').set_index('feature')['mean']
        fig,axes=plt.subplots(1,2,figsize=(4.8,2.55),sharey=True)
        axes[0].barh(range(8),sh.reindex(features),color='#294662')
        axes[1].barh(range(8),perm.reindex(features),color='#3b7f86')
        axes[0].set_yticks(range(8),[short.get(k,LABELS[k]) for k in features])
        axes[0].set(title='(a) Signed-grouped SHAP',xlabel='Mean absolute log-odds')
        axes[1].set(title='(b) Permutation',xlabel='Mean AP drop')
        axes[0].set_xlim(0,1.35);axes[1].set_xlim(0,.225)
        for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        fig.tight_layout(pad=.45,w_pad=.9);fig.savefig(FIG/'importance_compact.png',dpi=300,bbox_inches='tight');plt.close(fig)

        predictions=pd.read_csv(ROOT/'training_results/results/test_predictions.csv')
        fig,ax=plt.subplots(figsize=(4.8,2.7))
        colors=['#294662','#d78621','#357b6d','#9b4a58','#6c66ad']
        for color,name in zip(colors,['Logistic Regression','Random Forest','XGBoost','LightGBM','MLP']):
            p,r,_=precision_recall_curve(predictions.actual,predictions[name])
            ap=average_precision_score(predictions.actual,predictions[name])
            ax.plot(r,p,color=color,lw=1.2,label=f'{name} ({ap:.3f})')
        ax.axhline(predictions.actual.mean(),color='#777777',ls='--',lw=.8,label='Prevalence (0.123)')
        ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02))
        ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=7,framealpha=.95)
        fig.tight_layout(pad=.45);fig.savefig(FIG/'pr_compact.png',dpi=300,bbox_inches='tight');plt.close(fig)

        fig=plt.figure(figsize=(4.8,3.5));gs=fig.add_gridspec(2,2,height_ratios=[.9,1.4])
        top=fig.add_subplot(gs[0,:]);left=fig.add_subplot(gs[1,0]);right=fig.add_subplot(gs[1,1])
        top.axis('off');top.set_title('Dashboard: three evidence blocks',loc='left',fontweight='bold',fontsize=9)
        vals=[[r.Model,f'{r.PR_AUC:.3f}',f'{r.ROC_AUC:.3f}',f'{r.F1:.3f}'] for r in models.itertuples()]
        tbl=top.table(cellText=vals,colLabels=['Model','AP','ROC-AUC','F1 at 0.5'],loc='center',cellLoc='left',colWidths=[.44,.18,.2,.18])
        tbl.auto_set_font_size(False);tbl.set_fontsize(7.8);tbl.scale(1,1.05)
        for (i,j),cell in tbl.get_celld().items():cell.set_edgecolor('#dddddd');cell.set_facecolor('#e9eef3' if i==0 else 'white')
        f=sh.head(6).index[::-1];left.barh(range(6),sh.reindex(f),color='#294662')
        left.set_yticks(range(6),[short.get(k,LABELS[k]) for k in f],fontsize=7.5)
        left.set(title='Grouped SHAP',xlabel='Mean |SHAP| (log-odds)',xlim=(0,1.35))
        right.barh(range(5),abl.PR_AUC.iloc[::-1],color='#3b7f86')
        right.set_yticks(range(5),['F5','F4','F3','F2','F1'],fontsize=8)
        right.set(title='Ablation',xlabel='Held-out AP',xlim=(0,1))
        for i,v in enumerate(abl.PR_AUC.iloc[::-1]):right.text(v+.025,i,f'{v:.3f}',va='center',fontsize=7.2)
        for ax in [left,right]:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        fig.tight_layout(pad=.5,h_pad=1.2,w_pad=1.2)
        fig.savefig(FIG/'dashboard_compact.png',dpi=300,bbox_inches='tight');plt.close(fig)

def build():
    models=pd.read_csv(RES/'model_results.csv')
    abl=pd.read_csv(RES/'ablation_results.csv')
    cats=pd.read_csv(RES/'category_importance.csv')
    sh=pd.read_csv(RES/'shap_feature_importance.csv',index_col=0).iloc[:,0].sort_values(ascending=False)
    old=pd.read_csv(RES/'shap_aggregation_audit.csv',index_col=0)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # Scientific workflow diagram: explicit status of dashboard and interpretation data.
    fig,ax=plt.subplots(figsize=(6.4,5));ax.axis('off')
    blocks=[('1  Dataset and target','150,000 synthetic rows; exclude score and level'),
            ('2  Stratified split','120,000 train / 30,000 test; seed 42'),
            ('3  Training-fold pipeline','Encoding, scaling and optional SMOTE inside folds'),
            ('4  Model selection','Five classifiers; 5-fold training CV; primary metric AP'),
            ('5  Held-out analyses','Test metrics, paired bootstrap, SHAP, permutation, ablations'),
            ('6  Dashboard (HTML/PDF)','Verified SQL, signed-grouped SHAP and saved results')]
    for i,(a,b) in enumerate(blocks):
        y=.96-i*.162
        ax.text(.5,y,a+'\n'+b,ha='center',va='top',fontsize=10,bbox={'boxstyle':'round,pad=.45','fc':'#f4f6f8','ec':'#566a7a'})
        if i<5:ax.annotate('',xy=(.5,y-.145),xytext=(.5,y-.10),arrowprops={'arrowstyle':'->','color':'#566a7a'})
    fig.tight_layout();fig.savefig(FIG/'pipeline.png',dpi=220,bbox_inches='tight');plt.close(fig)
    # A printable three-block dashboard replaces the illegible six-block figure.
    fig,axes=plt.subplots(3,1,figsize=(6.7,8.4),gridspec_kw={'height_ratios':[1,1.15,1.1]})
    axes[0].axis('off'); axes[0].set_title('Burnout research dashboard',loc='left',fontweight='bold',fontsize=13)
    values=[[r.Model,f'{r.PR_AUC:.3f}',f'{r.ROC_AUC:.3f}',f'{r.F1:.3f}'] for r in models.itertuples()]
    t=axes[0].table(cellText=values,colLabels=['Model','AP','ROC-AUC','F1 at 0.5'],loc='center',cellLoc='left',colWidths=[.46,.16,.20,.18])
    t.auto_set_font_size(False);t.set_fontsize(10.5);t.scale(1,1.35)
    for (i,j),c in t.get_celld().items():
        c.set_edgecolor('#dddddd');c.set_facecolor('#e9eef3' if i==0 else 'white')
    top=sh.head(6).iloc[::-1]
    axes[1].barh([LABELS.get(k,k) for k in top.index],top,color='#294662')
    axes[1].set_title('Signed-grouped SHAP',loc='left',fontsize=12)
    axes[1].set_xlabel('Mean |sum of signed dummy SHAP| (log-odds)')
    labs=['All 23','No psychology (20)','Work + lifestyle (15)','No treatment (21)','No psychology/treatment (18)']
    axes[2].barh(labs[::-1],abl.PR_AUC.iloc[::-1],color='#294662')
    axes[2].set(xlim=(0,1),xlabel='Held-out average precision')
    axes[2].set_title('Controlled feature ablation',loc='left',fontsize=12)
    for y,v in enumerate(abl.PR_AUC.iloc[::-1]): axes[2].text(v+.015,y,f'{v:.3f}',va='center',fontsize=10)
    fig.text(.02,.008,'150,000 synthetic records; fixed test n=30,000. Model explanations are not causal evidence.',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1),h_pad=1.5)
    fig.savefig(FIG/'dashboard_print.png',dpi=250,bbox_inches='tight')
    fig.savefig(ROOT/'dashboard_submission/Burnout_Dashboard.pdf',bbox_inches='tight')
    plt.close(fig)
    compact_figures(models,abl,sh)
    for p in FIG.glob('*.png'):shutil.copy2(p,PF/p.name)
    # Reuse unaffected benchmark figures, clearly identified as original fitted models.
    for name in ['class_distribution.png','pr_curves.png','confusion_matrix.png','threshold_curve.png']:
        shutil.copy2(ROOT/'training_results/figures'/name,PF/name)

    mrows=rows(models,['Model','PR_AUC','ROC_AUC','F1','Precision','Recall'],{k:lambda x:f'{x:.3f}' for k in ['PR_AUC','ROC_AUC','F1','Precision','Recall']})
    mrows+=r'\midrule Patel et al. (XGBoost + SMOTE) & n/r & 0.960 & 0.708 & 0.716 & 0.701 \\'
    catrows=rows(cats,['Category','SHAP_share','Perm_drop_PR_AUC','Top_feature'],{'SHAP_share':lambda x:f'{100*x:.2f}\\%', 'Perm_drop_PR_AUC':lambda x:f'{x:.4f}', 'Top_feature':lambda x:tex(LABELS[x])})
    arows=[]
    anames=['F1: all features','F2: no psychological','F3: work + lifestyle','F4: no treatment','F5: no psychology/treatment']
    for i,r in abl.iterrows():
        arows.append(f"{tex(anames[i])} & {int(r.N_features)} & {r.PR_AUC:.3f} & [{r.AP_CI_low:.3f}, {r.AP_CI_high:.3f}] & {r.ROC_AUC:.3f} & {r.Drop_in_PR_AUC:.4f} \\\\")
    q1=pd.read_csv(ROOT/'sql_analysis/query_1_results.csv').set_index('grp')
    q3=pd.read_csv(ROOT/'sql_analysis/query_3_results.csv')
    heatrows=[]
    for i,lab in enumerate([r'$s<4$',r'$4\leq s<6$',r'$6\leq s<8$',r'$s\geq8$']):
        vals=q3.iloc[i*4:i*4+4]
        heatrows.append(lab+' & '+' & '.join(f'{r.pct_at_risk:.1f}\\% ({int(r.n):,})' for r in vals.itertuples())+r' \\')
    featuretable=[]
    from revision_analysis import CATEGORIES
    for c,fs in CATEGORIES.items():featuretable.append(tex(c)+' & '+str(len(fs))+' & '+', '.join(r'\texttt{'+tex(x)+'}' for x in fs)+r' \\')
    related=[
      r'Patel et al.\ \cite{patel} & 150,000 synthetic records; binary burnout; XGBoost + SMOTE & At Risk F1 0.7081, ROC-AUC 0.9604; AP not reported & Same dataset family; settings and exact test identities differ. \\',
      r'Adapa et al.\ \cite{adapa} & Healthcare staff survey: 450 invited, 59.3\% participation; two-item MBI; RF with 8 selected predictors & CV ROC-AUC 0.755; held-out ROC-AUC 0.811 & Real-data burnout classifier; different population, instrument and sample; AP not reported. \\',
      r'Trinkenreich et al.\ \cite{trink23} & Survey of 3,281 developers; structural equation modelling & Organizational resources and satisfaction associated with reduced burnout & Observational, single company; no shared burnout AP benchmark. \\',
      r'Trinkenreich et al.\ \cite{trink24} & 13,343 complete survey responses; 90-day employment status; ML & Tuned RF F-measure 0.880 for employees who left & Attrition outcome, not our burnout label; not a numerical performance comparator. \\',
      r'Feng et al.\ \cite{feng} & 442 developers; PLS-SEM, regression and qualitative responses & Associations among GenAI-related demands, resources and burnout & Observational self-report and a different task. \\',
      r'Ribeiro\ \cite{ribeiro} & 411 respondents; cross-sectional survey and factor analysis & Instability associated with exhaustion and cynicism & Convenience/self-selected sample; no classifier benchmark. \\',
      r'Michels et al.\ \cite{michels} & Qualitative study of two participants & Seven categories of developer overwhelm & Exploratory evidence; overwhelm is not a binary burnout diagnosis. \\']

    source=(PAPER/'manuscript_template.tex.in').read_text(encoding='utf-8')
    replacements={
        '@@RELATED@@':table('Numerical burnout-classification comparators; populations and outcomes differ.','tab:related',r'p{.18\textwidth}p{.31\textwidth}p{.41\textwidth}',r'Study & Data and method & Reported results and boundary',
            r'Patel et al.\ \cite{patel} & 150,000 synthetic records; XGBoost + SMOTE & F1 0.7081; ROC-AUC 0.9604; AP not reported. Related data, unmatched settings/test rows. \\'+'\n'+
            r'Adapa et al.\ \cite{adapa} & Healthcare; 450 invited, 59.3\% participation; RF, eight selected predictors & Five-fold CV ROC-AUC 0.755; test 0.811 (80/20 split). Different population/MBI label; AP not reported. \\'),
        '@@FEATURES@@':table('Predictor categories; social support consistently belongs to Work environment.','tab:features',r'p{.21\textwidth}rp{.66\textwidth}',r'Category & Count & Variables','\n'.join(featuretable)),
        '@@PIPELINE@@':figure('pipeline_compact.png','Implemented evaluation workflow; preprocessing and SMOTE fit only training folds.','fig:pipeline',width='1.0'),
        '@@PARAMS@@':table('Configured hyperparameters. JSON includes fitted defaults for four models and archived XGBoost training kwargs.','tab:params',r'p{.23\textwidth}p{.72\textwidth}',r'Model & Configuration','\n'.join([
            r'Logistic Regression & $C=1$, L2, lbfgs, tolerance $10^{-4}$, max\_iter=2000, balanced weights. \\',
            r'Random Forest & 300 trees, depth 8, min\_leaf=50, sqrt features, gini, bootstrap, balanced\_subsample, seed 42. \\',
            r'XGBoost & 300 trees, depth 3, rate 0.05, row/column sample 0.8, lambda=5, min\_child\_weight=10, binary:logistic, logloss, positive weight 7.1610, seed 42. \\',
            r'LightGBM & 300 trees, 8 leaves, rate 0.05, lambda=5, min\_child\_samples=200, row/column sample 0.8, subsample\_freq=1, positive weight 7.1610, seed 42. \\',
            r'MLP & (64,32), ReLU, Adam, rate 0.001, alpha=0.001, batch 200, max\_iter=100, early stopping, validation 0.1, patience 10, tolerance $10^{-4}$, seed 42; unweighted. \\'])),
        '@@HEAT@@':table(r'At Risk rates by stress $s$ and weekly hours $h$; counts in parentheses, all $n\geq30$.','tab:heat',r'lrrrr',r'Stress & $h\leq40$ & $40<h\leq50$ & $50<h\leq60$ & $h>60$','\n'.join(heatrows)),
        '@@CATEGORY@@':table('Category attribution from signed-grouped SHAP and joint permutation; shares are not causal effects.','tab:importance',r'p{.24\textwidth}rrp{.30\textwidth}',r'Category & SHAP share & AP drop & Top feature',catrows),
        '@@IMPORTANCE@@':figure('importance_compact.png','Original-variable importance: (a) signed-grouped SHAP on 2,000 test records; (b) AP drop on 10,000 test records, five permutations. Shared rows show the eight leading SHAP variables.','fig:importance',width='1.0'),
        '@@MODELS@@':table('At Risk test metrics on F1 (23 predictors). Our threshold is 0.5; published settings differ; n/r means not reported.','tab:models',r'p{.32\textwidth}rrrrr',r'Model & AP & ROC-AUC & F1 & Prec. & Rec.',mrows),
        '@@PR@@':figure('pr_compact.png','Precision--recall curves for the five retained fitted models on the fixed test set. Legend values are AP.','fig:pr',width='1.0'),
        '@@ABLATION@@':table('Controlled Logistic Regression ablations. Intervals use 2,000 paired bootstraps; AP loss is F1 minus the row, using unrounded values. F3 and F5 round to the same AP: 0.689199 and 0.689144 before rounding, respectively.','tab:ablation',r'p{.32\textwidth}rrlrr',r'Feature set & $p$ & AP & AP 95\% CI & ROC & AP loss','\n'.join(arows)),
        '@@DASHBOARD@@':figure('dashboard_compact.png',r'Three-block dashboard: metrics, signed-grouped SHAP and feature ablations F1--F5. Full-size interactive HTML and printable PDF accompany the project.','fig:dashboard',width='1.0'),
    }
    for marker,value in replacements.items():
        assert source.count(marker)==1,marker
        source=source.replace(marker,value)
    assert '@@' not in source and '\r' not in source
    refs={
      'who':r'World Health Organization: Burn-out an occupational phenomenon: International Classification of Diseases (2019). \url{https://www.who.int/news/item/28-05-2019-burn-out-an-occupational-phenomenon-international-classification-of-diseases}. Accessed 19 October 2026.',
      'wellness':r'Developer Nation: State of Developer Wellness Report 2024. Industry survey report (2024). \url{https://developernation.net/resources/reports/state-of-developer-wellness-report-2024/}. Accessed 27 October 2026.',
      'padmashali':r'Padmashali, A.: Prevalence of stress and burnout among information technology professionals during COVID-19 pandemic due to work from home situation. International Journal of Community Medicine and Public Health 10(4), 1530--1536 (2023). \url{https://doi.org/10.18203/2394-6040.ijcmph20230936}.',
      'trink23':r'Trinkenreich, B., Stol, K.-J., Steinmacher, I., Gerosa, M., Sarma, A., Lara, M., Feathers, M., Ross, N., Bishop, K.: A model for understanding and reducing developer burnout. In: ICSE Software Engineering in Practice (2023). \url{https://doi.org/10.1109/ICSE-SEIP58684.2023.00010}.',
      'trink24':r'Trinkenreich, B., Santos, F., Stol, K.-J.: Predicting attrition among software professionals: antecedents and consequences of burnout and engagement. ACM Trans. Softw. Eng. Methodol. 33(8), 218:1--218:45 (2024). \url{https://doi.org/10.1145/3691629}.',
      'mapping':r'Tulili, T.R., Capiluppi, A., Rastogi, A.: Burnout in software engineering: a systematic mapping study. Information and Software Technology 155, 107116 (2023). \url{https://doi.org/10.1016/j.infsof.2022.107116}.',
      'review':r'Tulili, T.R., Rastogi, A., Capiluppi, A.: Machine learning models for the early detection of burnout in software engineering: a systematic literature review. arXiv:2603.23063v1 (2026). \url{https://doi.org/10.48550/arXiv.2603.23063}.',
      'feng':r'Feng, Z., Afroz, S., Sarma, A.: From gains to strains: modeling developer burnout with GenAI adoption. In: Proceedings of the IEEE/ACM 48th International Conference on Software Engineering (ICSE-SEIS), pp. 125--136. ACM (2026). \url{https://doi.org/10.1145/3786581.3786934}.',
      'ribeiro':r'Ribeiro, D.M.: Understanding the relationships between the perceptions of burnout and instability in software engineering. In: Proceedings of the XXXVI Brazilian Symposium on Software Engineering, pp. 58--67. ACM (2022). \url{https://doi.org/10.1145/3555228.3555251}.',
      'michels':r'Michels, L.-M., Petkova, A., Richter, M., Farley, A., Graziotin, D., Wagner, S.: Overwhelmed software developers. IEEE Software 41(4), 51--59 (2024). \url{https://doi.org/10.1109/MS.2024.3374821}.',
      'un':r'United Nations, Department of Economic and Social Affairs: Goal 3: Ensure healthy lives and promote well-being for all at all ages. Target 3.4 (n.d.). \url{https://sdgs.un.org/goals/goal3}. Accessed 19 October 2026.',
      'patel':r'Patel, T., Sharma, G., Sarangi, P.K., Saxena, M., Dobariya, A., Ionescu, V.M., Bizon, N.: Feature attribution dynamics under imbalanced learning algorithms: an explainable machine learning framework for burnout risk prediction. Algorithms 19(9), 716 (2026). \url{https://doi.org/10.3390/a19090716}.',
      'adapa':r'Adapa, K., Pillai, M., Foster, M., Charguia, N., Mazur, L.: Using explainable supervised machine learning to predict burnout in healthcare professionals. Studies in Health Technology and Informatics 294, 58--62 (2022). \url{https://doi.org/10.3233/SHTI220396}.',
      'dataset':r'Gupta, S.: Employee Mental Health \& Burnout Dataset. Kaggle, version 1, released 15 March 2026 (2026). \url{https://www.kaggle.com/datasets/suhanigupta04/employee-mental-health-and-burnout-dataset}. Accessed 27 October 2026.',
      'rf':r'Breiman, L.: Random forests. Machine Learning 45(1), 5--32 (2001).',
      'xgb':r'Chen, T., Guestrin, C.: XGBoost: a scalable tree boosting system. In: ACM SIGKDD, pp. 785--794 (2016).',
      'lgb':r'Ke, G., et al.: LightGBM: a highly efficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems 30, 3146--3154 (2017).',
      'sklearn':r'Pedregosa, F., et al.: Scikit-learn: machine learning in Python. Journal of Machine Learning Research 12, 2825--2830 (2011).',
      'smote':r'Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P.: SMOTE: synthetic minority over-sampling technique. Journal of Artificial Intelligence Research 16, 321--357 (2002).',
      'saito':r'Saito, T., Rehmsmeier, M.: The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE 10(3), e0118432 (2015).',
      'shap':r'Lundberg, S.M., Lee, S.-I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems 30, 4765--4774 (2017).',
    }
    order=[]
    for group in re.findall(r'\\cite\{([^}]+)\}',source):
        for key in group.split(','):
            if key not in order:order.append(key)
    assert set(order)==set(refs)
    source+='\n'+r'\begin{thebibliography}{99}'+'\n'+'\n'.join(r'\bibitem{'+k+'} '+refs[k] for k in order)+'\n'+r'\end{thebibliography}\end{document}'
    # Break long code identifiers and URLs without shrinking the LNCS type size.
    source=re.sub(r'\\texttt\{([^}]+)\}',lambda m:r'\path{'+m.group(1).replace(r'\_','_')+'}',source)
    source=source.replace('p{.',r'>{\raggedright\arraybackslash}p{.')
    # Flush only at top-level section boundaries; per-subsection flushing was
    # creating isolated float pages and large empty areas in the old manuscript.
    source=re.sub(r'(?m)^(\\section\{)', r'\\FloatBarrier\n\1', source)
    source=source.replace('\\FloatBarrier\n\\FloatBarrier\n', '\\FloatBarrier\n')
    (PAPER/'main.tex').write_text(source,encoding='utf-8')
    # A separate bibliographic audit makes the recency window explicit.
    recent=['wellness','padmashali','trink23','trink24','mapping','review','feng','ribeiro','michels','patel','dataset','adapa']
    (RES/'reference_audit.json').write_text(json.dumps({'references':len(refs),'recent_2022_2026':len(recent),
                    'recent_fraction':len(recent)/len(refs),'recent_ids':recent,
                    'caveat':'Kaggle version-1 release date verified against publisher metadata saved in revision_20261022/reference_sources. Even excluding Kaggle, 11/20 sources fall in 2022-2026. No Scopus-indexing claim is made for preprints.'},indent=2))
    build_dashboard()
    print('Manuscript and dashboard sources regenerated:',PAPER/'main.tex')

def build_dashboard():
    d=json.loads((ROOT/'dashboard/data.json').read_text(encoding='utf-8'))
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Burnout research dashboard</title>
<style>:root{font:16px/1.5 system-ui;color:#192c3d;background:#f6f7f8}body{max-width:1250px;margin:auto;padding:32px}h1{line-height:1.15}h2{font-size:22px;margin-top:0}section{background:white;border:1px solid #d9dee3;margin:24px 0;padding:24px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #dde2e7}th{background:#edf1f5}small,.muted{color:#536273}select{font:inherit;padding:6px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}.scroll{overflow:auto}.bar{height:8px;background:#2c4b69}.label{display:flex;justify-content:space-between;gap:20px}article{margin:14px 0}.value{color:#213d57}@media(max-width:800px){body{padding:16px}.columns{display:block}section{padding:16px}}@media print{body{padding:0;background:white}section{break-inside:avoid}}</style>
<h1>Burnout risk classification research dashboard</h1><p>Group 15 · ADY201m · Paper update 27 October 2026 · Numerical analysis 19 October 2026</p><p class="muted">150,000 synthetic employee records. AP = average precision. These results describe a synthetic benchmark, not real-world or clinical validation.</p>
<label>Work mode <select id="mode"><option>All</option><option>Hybrid</option><option>Onsite</option><option>Remote</option></select></label><small> Filters descriptive summaries only. Model results remain on the fixed test set.</small>
<section><h2>Descriptive overview</h2><p id="summary"></p><div id="companies"></div></section>
<section><h2>Five-model held-out comparison</h2><div class="scroll" id="models"></div><small>Default threshold 0.5. AP intervals: 2,000 ordinary paired test-row bootstrap resamples, conditional on fitted models. MLP is an unweighted reference configuration.</small></section>
<div class="columns"><section><h2>Signed-grouped SHAP</h2><p class="muted">Sum signed dummy contributions within each observation, then take absolute values and average. Log-odds units; 2,000 test records; 100 effective background records from 1,000 training candidates.</p><div id="shap"></div></section><section><h2>Category attribution</h2><div id="categories"></div><small>Shares of original-feature attribution magnitude, not explained variance or causal effects. Permutation AP drops are not additive.</small></section></div>
<section><h2>Feature ablation and treatment sensitivity</h2><div class="scroll" id="ablation"></div><small>RQ3 is F1 to F2 only. F3 also removes demographics and treatment. F4/F5 are post hoc treatment-removal sensitivity analyses on the existing test set.</small></section>
<section><h2>Stress and weekly working hours</h2><div class="scroll" id="heat"></div><p>Exact check: 0 At Risk records among 12,749 profiles with stress below 4. This is consistent with rule-like synthetic labels; the generator formula remains unavailable.</p></section>
<section><h2>Fixed test confusion matrix</h2><p>Threshold 0.5: TN 23,199 · FP 3,125 · FN 294 · TP 3,382. Precision 0.520 · Recall 0.920 · F1 0.664.</p><p>Training OOF F1-optimal threshold 0.8145: test precision 0.706 · recall 0.748 · F1 0.727.</p></section>
<p class="muted">Sources: analysis CSVs in revision_20261019/results; verified SQL in sql_analysis. No external/temporal validation or subgroup analysis. SHAP and permutation importance are associations of a fitted model.</p>
<script id="research-data" type="application/json">''' + json.dumps(d,ensure_ascii=False).replace('</','<\\/')+'''</script>
<script>
const d=JSON.parse(document.getElementById('research-data').textContent),$=id=>document.getElementById(id),fmt=(x,n=3)=>Number(x).toFixed(n),label=s=>s.replaceAll('_',' ');
function table(id,headers,records){const t=document.createElement('table'),head=t.createTHead().insertRow();headers.forEach(v=>{let c=document.createElement('th');c.textContent=v;head.append(c)});const b=t.createTBody();records.forEach(row=>{let r=b.insertRow();row.forEach(v=>r.insertCell().textContent=v)});$(id).replaceChildren(t)}
function bars(id,items,max){$(id).replaceChildren();items.forEach(x=>{let a=document.createElement('article'),p=document.createElement('div');p.className='label';let n=document.createElement('span'),v=document.createElement('span');n.textContent=label(x.name);v.textContent=fmt(x.value);p.append(n,v);let b=document.createElement('div');b.className='bar';b.style.width=(100*x.value/max)+'%';a.append(p,b);$(id).append(a)})}
function describe(){let v=d.groups.find(g=>g.mode===$('mode').value);$('summary').textContent=`${v.n.toLocaleString()} records · ${v.risk.toLocaleString()} At Risk (${fmt(100*v.risk/v.n,2)}%) · mean weekly hours ${v.hours}`;table('companies',['Company size','Records','At Risk rate'],v.groups.map(g=>[g.name,g.n.toLocaleString(),fmt(100*g.risk/g.n,2)+'%']))}
$('mode').addEventListener('change',describe);describe();
table('models',['Model','AP (95% CI)','ROC-AUC','F1','Precision','Recall'],d.models.map(m=>[m.Model,fmt(m.PR_AUC)+' '+m.PR_AUC_CI,fmt(m.ROC_AUC),fmt(m.F1),fmt(m.Precision),fmt(m.Recall)]));
bars('shap',d.shap.slice(0,8),1.3);table('categories',['Category','SHAP share','Joint permutation AP drop'],d.categories.map(c=>[c.Category,fmt(100*c.SHAP_share,2)+'%',fmt(c.Perm_drop_PR_AUC,4)]));
table('ablation',['Feature set','Predictors','AP (95% CI)','ROC-AUC','AP loss vs F1 (95% CI)'],d.ablation.map(a=>[a.Feature_set,a.N_features,fmt(a.PR_AUC)+' ['+fmt(a.AP_CI_low)+', '+fmt(a.AP_CI_high)+']',fmt(a.ROC_AUC),fmt(a.Drop_in_PR_AUC,4)+' '+a.Drop_CI_95]));
const heat=[];for(let i=0;i<4;i++)heat.push([['Stress <4','4 to <6','6 to <8','Stress >=8'][i],...d.heat.slice(i*4,i*4+4).map(c=>fmt(c.pct_at_risk,1)+'% (n='+c.n.toLocaleString()+')')]);table('heat',['Stress / hours','<=40','>40 to 50','>50 to 60','>60'],heat);
</script></html>'''
    for p in [ROOT/'dashboard/burnout-dashboard.html',ROOT/'dashboard_submission/Burnout_Dashboard.html']:p.write_text(html,encoding='utf-8')
    shutil.copy2(FIG/'dashboard_print.png',ROOT/'dashboard_submission/Burnout_Dashboard.png')
    target=ROOT/'dashboard_submission/source_data';target.mkdir(exist_ok=True)
    for f in ['model_results.csv','shap_feature_importance.csv','category_importance.csv','ablation_results.csv','confusion_matrix_best.csv']:
        shutil.copy2(ROOT/'training_results/results'/f,target/f)
    for f in ['sql_queries.sql','query_1_results.csv','query_2_results.csv','query_3_results.csv']:shutil.copy2(ROOT/'sql_analysis'/f,target/f)
    shutil.copy2(ROOT/'dashboard/data.json',target/'dashboard_data.json')

if __name__=='__main__':build()
