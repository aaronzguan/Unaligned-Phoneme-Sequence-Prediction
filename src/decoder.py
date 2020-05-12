import torch
import Levenshtein as Lev
from ctcdecode import CTCBeamDecoder


class BeamCTCDecoder():
    def __init__(self, PHONEME_MAP, blank_index=0, beam_width=100):
        # Add the blank to the phoneme_map as the first element
        if PHONEME_MAP[blank_index] != ' ':
            PHONEME_MAP.insert(0, ' ')
        # Define the int_to_char dictionary
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(PHONEME_MAP)])
        self._decoder = CTCBeamDecoder(PHONEME_MAP, blank_id=blank_index, beam_width=beam_width, log_probs_input=True)

    def decode(self, probs, sizes=None):
        probs, sizes = probs.cpu(), sizes.cpu()
        out, _, _, seq_lens = self._decoder.decode(probs, sizes)
        # out: shape (batch_size, beam_width, seq_len)
        # seq_lens: shape (batch_size, beam_width)
        # The best sequences are indexed 0 in the beam_width dimension.
        strings = self.convert_to_strings(out[:, 0, :], seq_lens[:, 0])
        return strings

    def convert_to_strings(self, out, seq_len):
        """
        :param out: (batch_size, sequence_length)
        :param seq_len: (batch_size)
        :return:
        """
        out = out.cpu()
        results = []
        for b, utt in enumerate(out):
            size = seq_len[b]
            if size > 0:
                # Map each integer to the char using the int_to_char dictionary
                # Only get the original len and remove all the padding elements
                transcript = ''.join(map(lambda x: self.int_to_char[x.item()], utt[:size]))
            else:
                transcript = ''
            transcript = transcript.replace(' ', '')
            results.append(transcript)
        return results

    def Lev_dist(self, s1, s2):
        s1, s2 = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)