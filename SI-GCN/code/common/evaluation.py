
import numpy as np
from scipy import stats

class AccuracySummary:

    def __init__(self, p, r):
        self.results = {'MAE': None, # Mean Absolute Error
                        'MAPE': None,# Mean Absolute Percentage Error
                        'MSE': None, # Mean Square Error
                        'RMSE': None,# Root Mean Square Error
                        'CPC': None, # Common Part of Commuters
                        'SSI': None, # Sørensen similarity index
                        'SMC': None, # Spearman's rank correlation coefficient
                        'LLR': None} # Linear least-squares regression correlation coefficient
        self.p = p
        self.r = r
        p = np.array(p)
        r = np.array(r)

        #self.results['MAE'] = np.mean(np.abs(r-p))

        c1 = 0
        mape = 0
        c2 = 0
        ssi = 0
        for i in range(p.shape[0]):
            if r[i]:
                mape += np.abs((r[i]-p[i])/r[i])
                c1 += 1
            if r[i]+p[i]:
                ssi += min(r[i], p[i])/(r[i]+p[i])
                c2 += 1
        self.results['MAPE'] = mape/c1
        #self.results['SSI'] = ssi*2/(c2^2)

        #self.results['MSE'] = np.mean(np.square(r-p))
        self.results['RMSE'] = np.sqrt(np.mean(np.square(r-p)))

        stack = np.column_stack((p, r))
        self.results['CPC'] = 2 * np.sum(np.min(stack, axis=1)) / np.sum(stack)

        self.results['SMC'] = stats.spearmanr(r, p)
        #self.results['LLR'] = stats.linregress(r, p)

    def accuracy_string(self):
        return 'RMSE'

    def pretty_print(self):
        print("----------Evaluating accuracy for test triplets----------")
        print('proportion of zeros:', round(np.sum(self.p < 0.5) / self.p.shape[0], 3))
        print('real_min:', min(self.r), ', real_max:', max(self.r))
        print('pred_min:', int(min(self.p)), ', pred_max:', int(max(self.p)))
        print('real:', list(self.r[:20]))
        print('pred:', list(map(int, self.p[:20])))
        for item in self.results.items():
            if not item[1]:
                continue

            if item[0] == 'SMC':
                print('SMC: correlation =', round(item[1][0], 3), ', p-value =', round(item[1][1], 3))
            elif item[0] == 'LLR':
                print('LLR: R =', round(item[1][2], 3), ', p-value =', round(item[1][3], 3))
            else:
                print(item[0], end=': ')
                print(round(item[1], 3), end='\n')
        print("---------------------------------------------------------")

class AccuracyScore:

    def __init__(self, pred, real):
        self.pred = pred
        self.real = real

    def summarize(self):
        summary = self.get_summary()
        summary.pretty_print()

    def get_summary(self):
        return AccuracySummary(self.pred, self.real)

        
class Scorer:

    def __init__(self, settings, model, threshold):
        self.settings = settings
        self.model = model
        self.iter = 0
        self.RMSE = []
        self.SMC = []
        self.threshold = threshold

    def register_model(self, model):  #model_builder.build_decoder: BilinearDiag object inheriting from Model
        self.model = model

    def compute_accuracy_scores(self, triples, output=False):
        pred = self.model.score(triples) + self.threshold
        real = triples[:,3]
        score = AccuracyScore(pred, real)

        if output:
            self.dump_all_scores(pred, real)

        return score

    def compute_scores(self, triples, output=False):
        if self.settings['Metric'] == 'Accuracy':
            return self.compute_accuracy_scores(triples, output)

    def dump_all_scores(self, pred, real):
        #self.iter += 500
        #np.savetxt('../data/output/iter_'+str(self.iter)+'.txt', pred, delimiter=',')


        self.iter += 50   # change the corresponding self.iter
        if self.iter < 10051:
            self.RMSE.append(np.sqrt(np.mean(np.square(np.array(real)-np.array(pred)))))
            self.SMC.append(stats.spearmanr(np.array(real), np.array(pred))[0])
        if self.iter == 10050:
            np.savetxt('../data/output/GCN_RMSE_th30.txt', np.array(self.RMSE), fmt='%.3f', delimiter=',')
            np.savetxt('../data/output/GCN_SMC_th30.txt', np.array(self.SMC), fmt='%.3f', delimiter=',')




