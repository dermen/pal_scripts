# coding: utf-8

lines = open('taspase_maroon.stream').readlines()

geoms = []
chunks_xtals = []
chunks = []
in_geom=False
in_xtal=False
in_chunk=False
for l in lines:
    if 'Begin geom' in l:
        in_geom =True
        G = []
    elif 'End geom' in l:
        in_geom=False
        G.append(l)
        geoms.append(G.copy())
    elif 'Begin chun' in l:
        in_chunk=True
        CH = []
        xtals = []
    elif 'End chun' in l:
        in_chunk =False
        CH.append(l)
        chunks_xtals.append( xtals.copy())
        chunks.append(CH.copy())
    elif 'Begin crys' in l:
        in_xtal=True
        X = []
    elif 'End crys' in l:
        in_xtal=False
        X.append(l)
        xtals.append(X.copy())
        X = []
    if in_geom:
        G.append(l)
    if in_chunk and not in_xtal:
        CH.append(l)
    if in_xtal:
        X.append(l)

for latt in ["triclinic","monoclinic", "hexagonal"]:
    new_stream = open("%s.stream"%latt,'w')        
    header = "".join( lines[:3])
    new_stream.write(header)
    new_stream.write("".join(geoms[0]))
    for i,c in enumerate(chunks):
        chunk_s = "".join( c[:-1])
        new_stream.write(chunk_s)
        for x in chunks_xtals[i]:
            xs = "".join(x)
            if latt in xs:
                new_stream.write(xs)
        new_stream.write(c[-1])
    new_stream.close()


