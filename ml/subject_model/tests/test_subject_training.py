"""Participant-specific split and normalization tests, without cached EEG."""
import unittest
import numpy as np
from src.train_subject import partition
from src.data.preprocess import Trial

class SubjectTrainingTests(unittest.TestCase):
    def test_partition_and_training_only_statistics(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(144,1,22,500)); y=np.tile([0,1],72)
        trials=tuple(Trial(1,'0train','0',500+i*2000,750+i*2000,1250+i*2000,False) for i in range(144))
        groups,mean,std=partition(x,y,trials)
        again,_,_=partition(x,y,trials)
        self.assertEqual([len(groups[n].y) for n in ('train','validation')],[115,29])
        self.assertEqual(groups['train'].trials,again['train'].trials)
        self.assertFalse({t.id for t in groups['train'].trials}&{t.id for t in groups['validation'].trials})
        np.testing.assert_allclose(groups['train'].x.numpy().mean(axis=(0,3)),0,atol=1e-6)
        np.testing.assert_allclose(groups['train'].x.numpy().std(axis=(0,3)),1,atol=1e-6)
        ids={t.id:i for i,t in enumerate(trials)}
        idx=[ids[t.id] for t in groups['train'].trials]
        changed=x.copy()
        changed[[ids[t.id] for t in groups['validation'].trials]]+=1000
        _,mean2,std2=partition(changed,y,trials)
        np.testing.assert_array_equal(mean,mean2);np.testing.assert_array_equal(std,std2)
