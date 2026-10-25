"""Compare bundled native YuE2 tokenization with the published tiktoken ranks.

Requires regex and safetensors for validation only. Does not download assets.
"""
import argparse
import base64
from pathlib import Path
import subprocess
import unicodedata
import regex
from safetensors.torch import load_file

def main():
    p=argparse.ArgumentParser()
    for arg in ("model","probe","ranks","out"):p.add_argument("--"+arg,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ranks={base64.b64decode(token):int(rank) for token,rank in (line.split() for line in Path(a.ranks).read_bytes().splitlines())}
    pattern=regex.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    def encode(text):
        ids=[]
        for part in pattern.findall(unicodedata.normalize("NFC",text)):
            data=part.encode()
            if data in ranks:ids.append(ranks[data]);continue
            pieces=[bytes([b]) for b in data]
            while len(pieces)>1:
                candidates=[ranks.get(pieces[i]+pieces[i+1],10**9) for i in range(len(pieces)-1)]
                best=min(candidates)
                if best==10**9:break
                i=candidates.index(best);pieces[i:i+2]=[pieces[i]+pieces[i+1]]
            ids.extend(ranks[x] for x in pieces)
        return ids
    corpus=["Generate music with codec tokens from the given conditions.\n[Tags]\nGentle piano\n[Lyrics]\n[verse] Hello\n",
            "Cafe\u0301, déjà vu", "中文歌曲 日本語 가을", "\u1100\u1161\u11a8", "  A\n\nB\r\nC \t",
            "<|endoftext|> <|extra_0|> [chorus]", "X:1\nM:4/4\nL:1/8\nK:C\nCDEF GABc|", "Can't stop 12345! 😀"]
    for i,text in enumerate(corpus):
        source=out/"text.txt";source.write_text(text,encoding="utf-8",newline="")
        dest=out/"tokens.safetensors"
        subprocess.run([a.probe,"tokenize",a.model,str(source),str(dest)],check=True)
        actual=load_file(str(dest))["output"].int().tolist();expected=encode(text)
        assert actual==expected,(i,text,actual,expected)
    print(f"YuE2 tokenizer: {len(corpus)} published-rank comparisons passed")

if __name__=="__main__":main()
