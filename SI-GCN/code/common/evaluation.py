
import numpy as np
from scipy import stats

class AccuracySummary:

    def __init__(self, p, r):
        self.results = {'MAE': 0, # Mean Absolute Error
                        'MAPE': 0,# Mean Absolute Percentage Error
                        'MSE': 0, # Mean Square Error
                        'RMSE': 0,# Root Mean Square Error
                        'CPC': 0, # Common Part of Commuters
                        'SSI': 0, # Sørensen similarity index
                        'SMC': 0, # Spearman's rank correlation coefficient
                        'LLR': 0} # Linear least-squares regression correlation coefficient
        self.p = p
        self.r = r
        p = np.array(p)
        r = np.array(r)

        self.results['MAE'] = np.mean(np.abs(r-p))

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
        self.results['SSI'] = ssi*2/(c2^2)

        self.results['MSE'] = np.mean(np.square(r-p))
        self.results['RMSE'] = np.sqrt(self.results['MSE'])

        stack = np.column_stack((p, r))
        self.results['CPC'] = 2 * np.sum(np.min(stack, axis=1)) / np.sum(stack)

        self.results['SMC'] = stats.spearmanr(r, p)
        self.results['LLR'] = stats.linregress(r, p)

    def accuracy_string(self):
        return 'MSE'

    def pretty_print(self):
        print("----------Evaluating accuracy for test triplets----------")
        print('ratio of non-zeros', np.sum(self.p != 0) / self.p.shape[0])
        print('real_min:', min(self.r), ', real_max:', max(self.r))
        print('pred_min:', int(min(self.p)), ', pred_max:', int(max(self.p)))
        print('real:', list(self.r[:20]))
        print('pred:', list(map(int, self.p[:20])))
        for item in self.results.items():
            if item[0] == 'SMC':
                print('SMC: correlation =', round(item[1][0], 3), ', p-value =', round(item[1][1], 3))
            elif item[0] == 'LLR':
                print('LLR: R =', round(item[1][2], 3), ', p-value =', round(item[1][3], 3))
            else:
                print(item[0], end=': ')
                print(str(round(item[1],3)), end='\n')
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

    def __init__(self, settings, model, outpath):
        self.settings = settings
        self.model = model
        self.iter = 0
        self.RMSE = []
        self.SMC = []
        self.outpath = outpath

    def register_model(self, model):  #model_builder.build_decoder: BilinearDiag object inheriting from Model
        self.model = model

    def compute_accuracy_scores(self, triples, verbose=False):
        pred = self.model.score(triples[:,:3]) + 30
        real = triples[:,3]
        score = AccuracyScore(pred, real)

        if verbose:
            self.iter += 50
            if self.iter < 10000:
                self.RMSE.append(np.sqrt(np.mean(np.square(np.array(real)-np.array(pred)))))
                self.SMC.append(stats.spearmanr(np.array(real), np.array(pred))[0])
            elif self.iter == 10000:
                np.savetxt(self.outpath+'/results/RMSE.txt', np.array(self.RMSE), fmt='%.3f', delimiter=',')
                np.savetxt(self.outpath + '/results/SMC.txt', np.array(self.SMC), fmt='%.3f', delimiter=',')

        return score

    def compute_scores(self, triples, verbose=False):
        if self.settings['Metric'] == 'Accuracy':
            return self.compute_accuracy_scores(triples, verbose=verbose)

    def dump_all_scores(self, triples, filename):
        pass



