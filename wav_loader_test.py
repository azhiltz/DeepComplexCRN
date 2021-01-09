#coding: utf-8

import unittest as test
import wav_loader as loader 

dns_path="/Users/ltz/DataCard/AIDenoise/DNS-home/dataset"

class WavLoaderTest(test.TestCase):
    def testFindFile(self):
        noisy_path, clean_path = loader.get_all_file_name( dns_path )
        self.assertGreater( len(noisy_path), 1, "file number is larger than 1" )
        self.assertEqual( len(noisy_path), len(clean_path) )
    
    def testLoader(self):
        noisy_path, clean_path = loader.get_all_file_name( dns_path )
        WL = loader.WavDataset( noisy_path, clean_path )

        self.assertGreater( len(WL), 0 )

        s1, s2 = WL[0]
        self.assertIsNotNone( s1 )
        self.assertIsNotNone( s2 )

if __name__ == "__main__":
    test.main()