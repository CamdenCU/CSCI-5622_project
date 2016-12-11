from numpy import *
from util.gen_util import *
from util.math_util import *
from util.dtree_util import *
from rnn.adagrad import Adagrad
import rnn.propagation as prop
from classify.learn_classifiers import validate
import cPickle, time, argparse
from multiprocessing import Pool
import copy


class particle:
    gBestX = [] #should be static atomic by default
    gBestErr = Inf
    chi = .7298
    theta1 = 1.49618
    theta2 = 1.49618
    swarmSize = 0

    def __init__(self):
        self.pBestX = []
        self.pBestErr = Inf
        self.v = []
        self.x = []
        self.id = particle.swarmSize
        particle.swarmSize += 1

    def resetV(self,vec):
        #print(self.id)
        self.v = random.uniform(-1.0,1.0,len(vec))
        self.x = random.uniform(-2.0,2.0,len(vec))
        self.pBestX = self.x
        if len(particle.gBestX) != len(vec): particle.gBestX=vec

    def update(self,err):
        if err<self.pBestErr:
            self.pBestErr = err
            self.pBestX = self.x
        if self.pBestErr<particle.gBestErr:
            particle.gBestErr = self.pBestErr
            particle.gBestX = self.pBestX
            print 'Global best update: {:.3f}'.format(particle.gBestErr)

    @staticmethod
    def step(part):
        if len(particle.gBestX)!=len(part.pBestX):particle.gBestX = part.pBestX
        u1 = random.uniform(0,particle.theta1,len(part.x))
        u2 = random.uniform(0,particle.theta2,len(part.x))
        dif1 = (part.pBestX-part.x)
        dif2 = (particle.gBestX-part.x)
        oldV = part.v
        part.v = particle.chi*(oldV+ (u1*dif1)+ (u2*(dif2)))
        print 'id:{:<3d} {:= 6.3f} <- {:= 6.3f} + ({:= 6.3f}({:= 6.3f})+{:= 6.3f}({:= 6.3f}))'.format(part.id,part.v[0], oldV[0],u1[0],dif1[0],u2[0],dif2[0])
        part.x = part.x+part.v

    @staticmethod
    def resetG():
        particle.gBestX = []
        particle.gBestErr = Inf


# this function computes the objective / grad for each minibatch
def objective_and_grad(par_data):

    params, d, len_voc, rel_list = par_data[0]
    data = par_data[1]
    params = unroll_params(params, d, len_voc, rel_list)

    (rel_dict, Wv, b, L) = params

    error_sum = 0.0
    num_nodes = 0
    tree_size = 0

    # compute error and gradient for each tree in minibatch
    # also keep track of total number of nodes in minibatch
    for index, tree in enumerate(data):

        nodes = tree.get_nodes()
        for node in nodes:
            node.vec = L[:, node.ind].reshape( (d, 1))

        tree.ans_vec = L[:, tree.ans_ind].reshape( (d, 1))

        prop.forward_prop(params, tree, d)
        error_sum += tree.error()
        tree_size += len(nodes)


    return (error_sum, tree_size)


# train qanta and save model
if __name__ == '__main__':
    
    # command line arguments
    parser = argparse.ArgumentParser(description='QANTA: a question answering neural network \
                                     with trans-sentential aggregation')
    parser.add_argument('-data', help='location of dataset', default='data/hist_split')
    parser.add_argument('-We', help='location of word embeddings', default='data/hist_We')
    parser.add_argument('-d', help='word embedding dimension', type=int, default=100)
    parser.add_argument('-np', '--num_proc', help='number of cores to parallelize over', type=int, \
                        default=6)
    parser.add_argument('-lW', '--lambda_W', help='regularization weight for composition matrices', \
                        type=float, default=0.)
    parser.add_argument('-lWe', '--lambda_We', help='regularization weight for word embeddings', \
                        type=float, default=0.)
    parser.add_argument('-b', '--batch_size', help='adagrad minibatch size (ideal: 25 minibatches \
                        per epoch). for provided datasets, 272 for history and 341 for lit', type=int,\
                        default=68)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs, can also determine \
                         dynamically via validate method', type=int, default=30)
    parser.add_argument('-agr', '--adagrad_reset', help='reset sum of squared gradients after this many\
                         epochs', type=int, default=3)
    parser.add_argument('-v', '--do_val', help='check performance on dev set after this many\
                         epochs', type=int, default=5)
    parser.add_argument('-o', '--output', help='desired location of output model', \
                         default='models/hist_params')

    args = vars(parser.parse_args())
    pool = Pool(processes=args['num_proc'],maxtasksperchild=10)

    ## load data
    vocab, rel_list, ans_list, tree_dict = \
        cPickle.load(open(args['data'], 'rb'))

    # four total folds in this dataset: train, test, dev, and devtest
    train_trees = tree_dict['dev']

    # - since the dataset that we were able to release is fairly small, the
    #   test, dev, and devtest folds are tiny. feel free to validate over another
    #   combination of these folds if you wish.
    val_trees = tree_dict['dev']

    ans_list = array([vocab.index(ans) for ans in ans_list])

    # NOTE: it significantly helps both accuracy and training time to initialize
    #       word embeddings using something like Word2Vec. we have provided word2vec
    #       embeddings for both datasets. for other data, we strongly recommend
    #       using a similar smart initialization. you can also randomly initialize, although
    #       this generally results in slower convergence to a worse local minima
    orig_We = cPickle.load(open(args['We'], 'rb'))
    # orig_We = gen_rand_we(len(vocab), d)

    # regularization lambdas
    lambdas = [args['lambda_W'], args['lambda_We']]

    # output log and parameter file destinations
    param_file = args['output']
    log_file = param_file.split('_')[0] + '_log'

    print 'number of training sentences:', len(train_trees)
    print 'number of validation sentences:', len(val_trees)
    rel_list.remove('root')
    print 'number of dependency relations:', len(rel_list)

    ## remove incorrectly parsed sentences from data
    # print 'removing bad trees train...'
    bad_trees = []
    for ind, tree in enumerate(train_trees):

        if tree.get(0).is_word == 0:
            print tree.get_words(), ind
            bad_trees.append(ind)

    # pop bad trees, higher indices first
    # print 'removed ', len(bad_trees)
    for ind in bad_trees[::-1]:
        train_trees.pop(ind)

    # print 'removing bad trees val...'
    bad_trees = []
    for ind, tree in enumerate(val_trees):

        if tree.get(0).is_word == 0:
            # print tree.get_words(), ind
            bad_trees.append(ind)

    # pop bad trees, higher indices first
    # print 'removed ', len(bad_trees)
    for ind in bad_trees[::-1]:
        val_trees.pop(ind)

    # add vocab lookup to leaves / answer
    print 'adding lookup'
    for tree in train_trees:
        tree.ans_list = ans_list[ans_list != tree.ans_ind]

    # generate params / We
    params = gen_dtrnn_params(args['d'], rel_list)
    rel_list = params[0].keys()

    # add We matrix to params
    params += (orig_We, )
    del orig_We
    r = roll_params(params, rel_list)

    dim = r.shape[0]
    print 'parameter vector dimensionality:', dim

    log = open(log_file, 'w')

    
    min_error = float('inf')

    particles = [particle() for x in range(10)]
    [x.resetV(r) for x in particles]
    

    lstring = ''

    # create mini-batches
    random.shuffle(train_trees)
    batches = [train_trees[x : x + args['batch_size']] for x in xrange(0, len(train_trees),
                args['batch_size'])]
    for epoch in range(0, args['num_epochs']):

        for batch in batches:
            epoch_error = 0.0
            minibatches = [batch[x*len(batch)/args['num_proc']:(x+1)*len(batch)/args['num_proc']] for x in range(args['num_proc'])]
            batch_error = 0.0
            bStart = time.time()
            for x in range(10):
                for curParticle in particles:
                    particle.step(curParticle)
                    result = pool.map(objective_and_grad, [([curParticle.x, args['d'], len(vocab), rel_list],x) for x in minibatches])
                    batch_error = sum(x[0] for x in result)/sum(x[1] for x in result)
                    #print 'id:{:d}  {:<5f} {:<5f} {:<5f}'.format(curParticle.id, curParticle.x[0],curParticle.x[1],curParticle.x[2])
                    epoch_error += batch_error
                    curParticle.update(batch_error)
                    #particle.step(curParticle)
            print 'epoch:{:<3d}id:{:<3d} batchError:{:.3f} time:{:.3f}'.format(epoch,curParticle.id,batch_error,time.time()-bStart)


        

        # save parameters if the current model is better than previous best model
        if epoch_error < min_error:
            min_error = epoch_error
            print 'saving model...'
            params = unroll_params(particle.gBestX, args['d'], len(vocab), rel_list)
            cPickle.dump( ( params, vocab, rel_list), open(param_file, 'wb'))

        # check accuracy on validation set
        if epoch % args['do_val'] == 0 and epoch != 0:
            print 'validating...'
            params = unroll_params(particle.gBestX, args['d'], len(vocab), rel_list)
            train_acc, val_acc = validate([train_trees, val_trees], params, args['d'])
            lstring = 'train acc = ' + str(train_acc) + ', val acc = ' + str(val_acc) + '\n\n\n'
            print lstring
            log.write(lstring)
            log.flush()

    log.close()