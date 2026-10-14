import json, re, sys
LAYOUT="g111684f229f_1_101"
PT="PT"
BLUE={"red":64/255,"green":124/255,"blue":186/255}
RED={"red":192/255,"green":57/255,"blue":75/255}
GREY={"red":0.4,"green":0.4,"blue":0.4}
HL={"red":1.0,"green":0.95,"blue":0.75}
HDR={"red":0.90,"green":0.90,"blue":0.90}

def parse(md):
    """**bold**, {b:text} blue, {r:text} red -> plain, list of (start,end,style)"""
    out=""; spans=[]
    i=0
    pat=re.compile(r"\*\*(.+?)\*\*|\{b:(.+?)\}|\{r:(.+?)\}|\{i:(.+?)\}")
    pos=0
    for m in pat.finditer(md):
        out+=md[pos:m.start()]
        s=len(out)
        if m.group(1) is not None: t=m.group(1); st={"bold":True}
        elif m.group(2) is not None: t=m.group(2); st={"bold":True,"foregroundColor":{"opaqueColor":{"rgbColor":BLUE}}}
        elif m.group(3) is not None: t=m.group(3); st={"bold":True,"foregroundColor":{"opaqueColor":{"rgbColor":RED}}}
        else: t=m.group(4); st={"italic":True}
        out+=t; spans.append((s,len(out),st)); pos=m.end()
    out+=md[pos:]
    return out,spans

def style_req(oid,s,e,st,cell=None):
    fields=",".join(st.keys())
    r={"updateTextStyle":{"objectId":oid,"textRange":{"type":"FIXED_RANGE","startIndex":s,"endIndex":e},"style":st,"fields":fields}}
    if cell: r["updateTextStyle"]["cellLocation"]=cell
    return r

def slide(sid, idx, title, body=None, size=18, table=None, source=None, body_box=None, image=None, extra_text=None):
    R=[]
    t_id=f"{sid}_title"; b_id=f"{sid}_body"
    R.append({"createSlide":{"objectId":sid,"insertionIndex":idx,"slideLayoutReference":{"layoutId":LAYOUT},
        "placeholderIdMappings":[{"layoutPlaceholder":{"type":"TITLE","index":0},"objectId":t_id},
                                 {"layoutPlaceholder":{"type":"BODY","index":1},"objectId":b_id}]}})
    R.append({"insertText":{"objectId":t_id,"text":title}})
    if body:
        lines=body.split("\n"); ind=[l.startswith("> ") for l in lines]
        body="\n".join(l[2:] if l.startswith("> ") else l for l in lines)
        txt,spans=parse(body)
        R.append({"insertText":{"objectId":b_id,"text":txt}})
        R.append(style_req(b_id,0,len(txt),{"fontSize":{"magnitude":size,"unit":PT},"bold":False,"fontFamily":"Arial"}))
        R.append({"updateParagraphStyle":{"objectId":b_id,"textRange":{"type":"ALL"},"style":{"lineSpacing":100,"spaceAbove":{"magnitude":0,"unit":PT},"spaceBelow":{"magnitude":6,"unit":PT}},"fields":"lineSpacing,spaceAbove,spaceBelow"}})
        for s,e,st in spans: R.append(style_req(b_id,s,e,st))
        pos=0
        for l,flag in zip(txt.split("\n"),ind):
            if flag:
                R.append({"updateParagraphStyle":{"objectId":b_id,"textRange":{"type":"FIXED_RANGE","startIndex":pos,"endIndex":pos+len(l)+1 if pos+len(l)<len(txt) else pos+len(l)},"style":{"indentStart":{"magnitude":28,"unit":PT},"indentFirstLine":{"magnitude":28,"unit":PT}},"fields":"indentStart,indentFirstLine"}})
                R.append(style_req(b_id,pos,pos+len(l),{"fontSize":{"magnitude":size-2,"unit":PT}}) if l else None)
            pos+=len(l)+1
        R[:]=[r for r in R if r is not None]
        if body_box:
            x,y,w,h=body_box
            R.append({"updatePageElementTransform":{"objectId":b_id,"applyMode":"ABSOLUTE","transform":{"scaleX":w/236.22,"scaleY":h/236.22,"translateX":x,"translateY":y,"shearX":0,"shearY":0,"unit":PT}}})
    else:
        R.append({"deleteObject":{"objectId":b_id}})
    if table:
        tb=f"{sid}_tbl"; rows=table["rows"]; nr=len(rows); nc=len(rows[0])
        x,y,w,h=table["box"]
        R.append({"createTable":{"objectId":tb,"rows":nr,"columns":nc,"elementProperties":{"pageObjectId":sid,
            "size":{"width":{"magnitude":w,"unit":PT},"height":{"magnitude":h,"unit":PT}},
            "transform":{"scaleX":1,"scaleY":1,"translateX":x,"translateY":y,"shearX":0,"shearY":0,"unit":PT}}}})
        fs=table.get("size",16)
        for ri,row in enumerate(rows):
            for ci,cell in enumerate(row):
                if cell=="": continue
                txt,spans=parse(cell)
                loc={"rowIndex":ri,"columnIndex":ci}
                R.append({"insertText":{"objectId":tb,"cellLocation":loc,"text":txt}})
                st={"fontSize":{"magnitude":fs,"unit":PT},"fontFamily":"Arial","bold":ri==0}
                R.append(style_req(tb,0,len(txt),st,loc))
                for s,e,sst in spans: R.append(style_req(tb,s,e,sst,loc))
        R.append({"updateTableCellProperties":{"objectId":tb,"tableRange":{"location":{"rowIndex":0,"columnIndex":0},"rowSpan":1,"columnSpan":nc},
            "tableCellProperties":{"tableCellBackgroundFill":{"solidFill":{"color":{"rgbColor":HDR}}},"contentAlignment":"MIDDLE"},"fields":"tableCellBackgroundFill,contentAlignment"}})
        for hr in table.get("highlight",[]):
            R.append({"updateTableCellProperties":{"objectId":tb,"tableRange":{"location":{"rowIndex":hr,"columnIndex":0},"rowSpan":1,"columnSpan":nc},
                "tableCellProperties":{"tableCellBackgroundFill":{"solidFill":{"color":{"rgbColor":HL}}}},"fields":"tableCellBackgroundFill"}})
        if "colw" in table:
            for ci,cw in enumerate(table["colw"]):
                R.append({"updateTableColumnProperties":{"objectId":tb,"columnIndices":[ci],"tableColumnProperties":{"columnWidth":{"magnitude":cw,"unit":PT}},"fields":"columnWidth"}})
    if extra_text:
        for k,(md,box,fs) in enumerate(extra_text):
            eid=f"{sid}_tx{k}"; x,y,w,h=box
            R.append({"createShape":{"objectId":eid,"shapeType":"TEXT_BOX","elementProperties":{"pageObjectId":sid,
                "size":{"width":{"magnitude":w,"unit":PT},"height":{"magnitude":h,"unit":PT}},
                "transform":{"scaleX":1,"scaleY":1,"translateX":x,"translateY":y,"shearX":0,"shearY":0,"unit":PT}}}})
            txt,spans=parse(md)
            R.append({"insertText":{"objectId":eid,"text":txt}})
            R.append(style_req(eid,0,len(txt),{"fontSize":{"magnitude":fs,"unit":PT},"fontFamily":"Arial","bold":False}))
            R.append({"updateParagraphStyle":{"objectId":eid,"textRange":{"type":"ALL"},"style":{"lineSpacing":100,"spaceBelow":{"magnitude":4,"unit":PT}},"fields":"lineSpacing,spaceBelow"}})
            for s,e,st in spans: R.append(style_req(eid,s,e,st))
    if image:
        url,box=image; x,y,w,h=box
        R.append({"createImage":{"objectId":f"{sid}_img","url":url,"elementProperties":{"pageObjectId":sid,
            "size":{"width":{"magnitude":w,"unit":PT},"height":{"magnitude":h,"unit":PT}},
            "transform":{"scaleX":1,"scaleY":1,"translateX":x,"translateY":y,"shearX":0,"shearY":0,"unit":PT}}}})
    if source:
        sid2=f"{sid}_src"
        R.append({"createShape":{"objectId":sid2,"shapeType":"TEXT_BOX","elementProperties":{"pageObjectId":sid,
            "size":{"width":{"magnitude":670,"unit":PT},"height":{"magnitude":22,"unit":PT}},
            "transform":{"scaleX":1,"scaleY":1,"translateX":25,"translateY":360,"shearX":0,"shearY":0,"unit":PT}}}})
        txt,spans=parse(source)
        R.append({"insertText":{"objectId":sid2,"text":txt}})
        R.append(style_req(sid2,0,len(txt),{"fontSize":{"magnitude":9,"unit":PT},"fontFamily":"Arial","foregroundColor":{"opaqueColor":{"rgbColor":GREY}}}))
        for s,e,st in spans: R.append(style_req(sid2,s,e,st))
    return R
