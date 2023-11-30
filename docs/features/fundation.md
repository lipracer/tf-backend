- python打包   
  将python打包目录从build/lib默认的目录移动到其他目录


- tensor支持layout、offset、stride属性  
  后面可以支持view相关操作

- host && device 内存泄漏检查

- GraphDef attribute支持，目前还未lower下来

- fusion pass支持（eager fusion）

- GraphDef转到mlir图

- 图模式dynamic shape
  - grpah 缓存算法 以及 将`shape`抽象成`symbolic shape`
  - runtime进行内存分配