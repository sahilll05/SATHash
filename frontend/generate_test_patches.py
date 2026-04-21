import numpy as np, struct, os

rng = np.random.default_rng(42)

def _p(fmt,*a): return struct.pack('<'+fmt,*a)

def write_geotiff_10band(filepath, arr, lat=20.0, lon=78.0):
    H,W,C = arr.shape
    assert C==10 and arr.dtype==np.uint16
    raw = arr.tobytes()
    bps   = b''.join(_p('H',16) for _ in range(10))
    sfmt  = b''.join(_p('H',1)  for _ in range(10))
    sw    = b'SatHash-TestPatch\x00'
    geo_scale = b''.join(_p('d',v) for v in [0.0001,0.0001,0.0])
    tiepoint  = b''.join(_p('d',v) for v in [0,0,0,lon,lat,0])
    geokeys   = b''.join(_p('H',v) for v in [1,1,0,3,1024,0,1,2,1025,0,1,1,2048,0,1,4326])

    ENTRIES=[]; EXTRA=bytearray()
    SHORT=3; LONG=4; DOUBLE=12; ASCII=2

    def inl(tag,typ,count,val):
        v = _p('HH',val,0) if typ==SHORT else _p('I',val)
        ENTRIES.append((tag,typ,count,v,None))
    def ext(tag,typ,count,blob):
        ENTRIES.append((tag,typ,count,None,blob))

    inl(256,SHORT,1,W); inl(257,SHORT,1,H)
    ext(258,SHORT,10,bps)
    inl(259,SHORT,1,1); inl(262,SHORT,1,1)
    inl(273,LONG,1,0)   # StripOffsets placeholder
    inl(277,SHORT,1,10); inl(278,SHORT,1,H)
    inl(279,LONG,1,len(raw)); inl(284,SHORT,1,1)
    ext(285,ASCII,len(sw),sw)
    ext(33550,DOUBLE,3,geo_scale)
    ext(33922,DOUBLE,6,tiepoint)
    ext(34735,SHORT,16,geokeys)
    ext(339,SHORT,10,sfmt)

    N=len(ENTRIES)
    IFD_SZ = 2+N*12+4
    IFD_ST = 8
    EX_ST  = IFD_ST+IFD_SZ
    extra_blob=bytearray()
    resolved=[]
    for tag,typ,count,val,blob in ENTRIES:
        if blob is None:
            resolved.append((tag,typ,count,val))
        else:
            off=EX_ST+len(extra_blob); extra_blob+=blob
            resolved.append((tag,typ,count,_p('I',off)))
    IMG_ST=EX_ST+len(extra_blob)
    patched=[]
    for tag,typ,count,val in resolved:
        if tag==273: val=_p('I',IMG_ST)
        patched.append((tag,typ,count,val))
    ifd=_p('H',N)
    for tag,typ,count,val in patched:
        ifd+=_p('HHI',tag,typ,count)+val
    ifd+=_p('I',0)
    with open(filepath,'wb') as f:
        f.write(b'II'+_p('H',42)+_p('I',IFD_ST)+ifd+extra_blob+raw)

def sf(fn, base, H=120, W=120): return fn(float(base),H,W)

def urban(b,H=120,W=120):
    yy,xx=np.mgrid[:H,:W]
    roads=np.clip((xx%15<2).astype(float)+(yy%15<2).astype(float),0,1)
    return b*(0.8+0.35*roads)+rng.normal(0,b*0.04,(H,W))

def forest(b,H=120,W=120):
    out=np.full((H,W),b)
    for _ in range(28):
        cx,cy=rng.integers(5,W-5),rng.integers(5,H-5)
        r=rng.integers(4,12); yy,xx=np.ogrid[:H,:W]
        out[((xx-cx)**2+(yy-cy)**2)<=r**2]=b*rng.uniform(0.85,1.2)
    return out+rng.normal(0,b*0.05,(H,W))

def water(b,H=120,W=120):
    x=np.linspace(0,6*np.pi,W); y=np.linspace(0,6*np.pi,H)
    rip=np.sin(x)[None,:]*0.06+np.cos(y)[:,None]*0.04
    out=b*(1+rip)+rng.normal(0,b*0.015,(H,W))
    for _ in range(5):
        cx,cy=rng.integers(20,W-20),rng.integers(20,H-20)
        yy,xx=np.ogrid[:H,:W]
        out[((xx-cx)**2+(yy-cy)**2)<rng.integers(100,400)]=b*rng.uniform(1.4,2.2)
    return out

def fields(b,H=120,W=120):
    out=np.full((H,W),b)
    for _ in range(rng.integers(5,12)):
        x0,y0=rng.integers(0,W-15),rng.integers(0,H-15)
        x1,y1=min(x0+rng.integers(15,45),W),min(y0+rng.integers(15,45),H)
        out[y0:y1,x0:x1]=b*rng.uniform(0.6,1.5)
    return out+rng.normal(0,b*0.04,(H,W))

def desert(b,H=120,W=120):
    x=np.linspace(0,3*np.pi,W); y=np.linspace(0,3*np.pi,H)
    dune=(np.sin(x*0.7+0.5)[None,:]+ np.cos(y*0.4+1.2)[:,None])*0.1
    return b*(1+dune)+rng.normal(0,b*0.018,(H,W))

def wetland(b,H=120,W=120):
    out=np.full((H,W),b)
    for _ in range(rng.integers(5,12)):
        cx,cy=rng.integers(10,W-10),rng.integers(10,H-10)
        r=rng.integers(6,18); yy,xx=np.ogrid[:H,:W]
        out[((xx-cx)**2+(yy-cy)**2)<r**2]=b*rng.uniform(0.25,0.55)
    for _ in range(rng.integers(4,9)):
        cx,cy=rng.integers(10,W-10),rng.integers(10,H-10)
        r=rng.integers(5,14); yy,xx=np.ogrid[:H,:W]
        out[((xx-cx)**2+(yy-cy)**2)<r**2]=b*rng.uniform(1.1,1.45)
    return out+rng.normal(0,b*0.03,(H,W))

TX={'urban':urban,'forest':forest,'water':water,'fields':fields,'desert':desert,'wetland':wetland}

def make_patch(profile,tex,H=120,W=120):
    return np.stack([
        np.clip(TX[tex](float(v),H,W),0,10000).astype(np.uint16)
        for v in profile], axis=-1)

PATCHES={
  "dense_urban_delhi":dict(
    profile=[1200,1400,1500,1600,1800,1900,2100,2000,2800,2200],
    texture="urban",lat=28.61,lon=77.20,
    desc="Dense urban — New Delhi, India"),
  "tropical_forest_ghats":dict(
    profile=[380,640,470,2600,4100,4700,5400,5100,1100,580],
    texture="forest",lat=13.50,lon=75.80,
    desc="Tropical rainforest — Western Ghats, Karnataka, India"),
  "ganges_river_delta":dict(
    profile=[920,1150,780,490,390,370,340,330,390,290],
    texture="water",lat=21.90,lon=88.80,
    desc="River delta / turbid water — Ganges Delta, West Bengal, India"),
  "dryland_agriculture_deccan":dict(
    profile=[780,900,1380,2050,2650,2850,3050,2920,2520,1820],
    texture="fields",lat=18.20,lon=76.50,
    desc="Dryland rain-fed agriculture — Deccan Plateau, Maharashtra, India"),
  "thar_desert_rajasthan":dict(
    profile=[3100,3400,3750,3580,3380,3180,2980,2860,3550,2980],
    texture="desert",lat=26.90,lon=70.90,
    desc="Barren sand desert — Thar Desert, Rajasthan, India"),
  "coastal_wetland_chilika":dict(
    profile=[600,880,690,1820,2450,2680,2850,2650,1420,910],
    texture="wetland",lat=19.75,lon=85.32,
    desc="Coastal wetland / mangrove — Chilika Lake, Odisha, India"),
}

out_dir="/home/claude/test_patches"
os.makedirs(out_dir,exist_ok=True)

print("="*65)
print("  SatHash — Sentinel-2 Style Test Patches  |  India")
print("="*65)
for name,cfg in PATCHES.items():
    arr=make_patch(cfg["profile"],cfg["texture"])
    path=os.path.join(out_dir,f"{name}.tif")
    write_geotiff_10band(path,arr,lat=cfg["lat"],lon=cfg["lon"])
    sz=os.path.getsize(path)/1024
    nir=cfg["profile"][6]; red=cfg["profile"][2]
    ndvi=(nir-red)/(nir+red+1e-6)
    print(f"  ✓  {name}.tif  ({sz:.0f} KB)  NDVI={ndvi:+.2f}")
    print(f"     {cfg['desc']}")
    print(f"     Coords: {cfg['lat']}N  {cfg['lon']}E")
    print()

print("Done! test_patches/ created.")
