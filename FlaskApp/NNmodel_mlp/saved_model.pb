??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.42v2.6.3-62-g9ef160463d18¶
v
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*
shared_namelayer1/kernel
o
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes

://*
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
:/*
dtype0
v
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/d*
shared_namelayer2/kernel
o
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes

:/d*
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:d*
dtype0
v
layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namelayer3/kernel
o
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel*
_output_shapes

:dd*
dtype0
n
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namelayer3/bias
g
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes
:d*
dtype0
v
layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d/*
shared_namelayer4/kernel
o
!layer4/kernel/Read/ReadVariableOpReadVariableOplayer4/kernel*
_output_shapes

:d/*
dtype0
n
layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namelayer4/bias
g
layer4/bias/Read/ReadVariableOpReadVariableOplayer4/bias*
_output_shapes
:/*
dtype0
v
layer5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*
shared_namelayer5/kernel
o
!layer5/kernel/Read/ReadVariableOpReadVariableOplayer5/kernel*
_output_shapes

://*
dtype0
n
layer5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namelayer5/bias
g
layer5/bias/Read/ReadVariableOpReadVariableOplayer5/bias*
_output_shapes
:/*
dtype0
v
layer6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*
shared_namelayer6/kernel
o
!layer6/kernel/Read/ReadVariableOpReadVariableOplayer6/kernel*
_output_shapes

://*
dtype0
n
layer6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namelayer6/bias
g
layer6/bias/Read/ReadVariableOpReadVariableOplayer6/bias*
_output_shapes
:/*
dtype0
v
layer7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*
shared_namelayer7/kernel
o
!layer7/kernel/Read/ReadVariableOpReadVariableOplayer7/kernel*
_output_shapes

://*
dtype0
n
layer7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namelayer7/bias
g
layer7/bias/Read/ReadVariableOpReadVariableOplayer7/bias*
_output_shapes
:/*
dtype0
v
layer8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*
shared_namelayer8/kernel
o
!layer8/kernel/Read/ReadVariableOpReadVariableOplayer8/kernel*
_output_shapes

:/*
dtype0
n
layer8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer8/bias
g
layer8/bias/Read/ReadVariableOpReadVariableOplayer8/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*%
shared_nameAdam/layer1/kernel/m
}
(Adam/layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/m*
_output_shapes

://*
dtype0
|
Adam/layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer1/bias/m
u
&Adam/layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/m*
_output_shapes
:/*
dtype0
?
Adam/layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/d*%
shared_nameAdam/layer2/kernel/m
}
(Adam/layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/m*
_output_shapes

:/d*
dtype0
|
Adam/layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/layer2/bias/m
u
&Adam/layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/m*
_output_shapes
:d*
dtype0
?
Adam/layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*%
shared_nameAdam/layer3/kernel/m
}
(Adam/layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/m*
_output_shapes

:dd*
dtype0
|
Adam/layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/layer3/bias/m
u
&Adam/layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/m*
_output_shapes
:d*
dtype0
?
Adam/layer4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d/*%
shared_nameAdam/layer4/kernel/m
}
(Adam/layer4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer4/kernel/m*
_output_shapes

:d/*
dtype0
|
Adam/layer4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer4/bias/m
u
&Adam/layer4/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer4/bias/m*
_output_shapes
:/*
dtype0
?
Adam/layer5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*%
shared_nameAdam/layer5/kernel/m
}
(Adam/layer5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer5/kernel/m*
_output_shapes

://*
dtype0
|
Adam/layer5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer5/bias/m
u
&Adam/layer5/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer5/bias/m*
_output_shapes
:/*
dtype0
?
Adam/layer6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*%
shared_nameAdam/layer6/kernel/m
}
(Adam/layer6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer6/kernel/m*
_output_shapes

://*
dtype0
|
Adam/layer6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer6/bias/m
u
&Adam/layer6/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer6/bias/m*
_output_shapes
:/*
dtype0
?
Adam/layer7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*%
shared_nameAdam/layer7/kernel/m
}
(Adam/layer7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer7/kernel/m*
_output_shapes

://*
dtype0
|
Adam/layer7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer7/bias/m
u
&Adam/layer7/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer7/bias/m*
_output_shapes
:/*
dtype0
?
Adam/layer8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*%
shared_nameAdam/layer8/kernel/m
}
(Adam/layer8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer8/kernel/m*
_output_shapes

:/*
dtype0
|
Adam/layer8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer8/bias/m
u
&Adam/layer8/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer8/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*%
shared_nameAdam/layer1/kernel/v
}
(Adam/layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/v*
_output_shapes

://*
dtype0
|
Adam/layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer1/bias/v
u
&Adam/layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/v*
_output_shapes
:/*
dtype0
?
Adam/layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/d*%
shared_nameAdam/layer2/kernel/v
}
(Adam/layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/v*
_output_shapes

:/d*
dtype0
|
Adam/layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/layer2/bias/v
u
&Adam/layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/v*
_output_shapes
:d*
dtype0
?
Adam/layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*%
shared_nameAdam/layer3/kernel/v
}
(Adam/layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/v*
_output_shapes

:dd*
dtype0
|
Adam/layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/layer3/bias/v
u
&Adam/layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/v*
_output_shapes
:d*
dtype0
?
Adam/layer4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d/*%
shared_nameAdam/layer4/kernel/v
}
(Adam/layer4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer4/kernel/v*
_output_shapes

:d/*
dtype0
|
Adam/layer4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer4/bias/v
u
&Adam/layer4/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer4/bias/v*
_output_shapes
:/*
dtype0
?
Adam/layer5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*%
shared_nameAdam/layer5/kernel/v
}
(Adam/layer5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer5/kernel/v*
_output_shapes

://*
dtype0
|
Adam/layer5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer5/bias/v
u
&Adam/layer5/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer5/bias/v*
_output_shapes
:/*
dtype0
?
Adam/layer6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*%
shared_nameAdam/layer6/kernel/v
}
(Adam/layer6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer6/kernel/v*
_output_shapes

://*
dtype0
|
Adam/layer6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer6/bias/v
u
&Adam/layer6/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer6/bias/v*
_output_shapes
:/*
dtype0
?
Adam/layer7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*%
shared_nameAdam/layer7/kernel/v
}
(Adam/layer7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer7/kernel/v*
_output_shapes

://*
dtype0
|
Adam/layer7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*#
shared_nameAdam/layer7/bias/v
u
&Adam/layer7/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer7/bias/v*
_output_shapes
:/*
dtype0
?
Adam/layer8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*%
shared_nameAdam/layer8/kernel/v
}
(Adam/layer8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer8/kernel/v*
_output_shapes

:/*
dtype0
|
Adam/layer8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer8/bias/v
u
&Adam/layer8/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?N
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?M
value?MB?M B?M
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
h

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
h

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
h

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
h

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
?
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemvmwmxmymzm{!m|"m}'m~(m-m?.m?3m?4m?9m?:m?v?v?v?v?v?v?!v?"v?'v?(v?-v?.v?3v?4v?9v?:v?
v
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15
 
v
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15
?

Dlayers
Elayer_regularization_losses
Fmetrics

trainable_variables
Gnon_trainable_variables
regularization_losses
	variables
Hlayer_metrics
 
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Ilayers
Jlayer_regularization_losses
Kmetrics
trainable_variables
Lnon_trainable_variables
regularization_losses
	variables
Mlayer_metrics
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Nlayers
Olayer_regularization_losses
Pmetrics
trainable_variables
Qnon_trainable_variables
regularization_losses
	variables
Rlayer_metrics
YW
VARIABLE_VALUElayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Slayers
Tlayer_regularization_losses
Umetrics
trainable_variables
Vnon_trainable_variables
regularization_losses
	variables
Wlayer_metrics
YW
VARIABLE_VALUElayer4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
?

Xlayers
Ylayer_regularization_losses
Zmetrics
#trainable_variables
[non_trainable_variables
$regularization_losses
%	variables
\layer_metrics
YW
VARIABLE_VALUElayer5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
?

]layers
^layer_regularization_losses
_metrics
)trainable_variables
`non_trainable_variables
*regularization_losses
+	variables
alayer_metrics
YW
VARIABLE_VALUElayer6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
?

blayers
clayer_regularization_losses
dmetrics
/trainable_variables
enon_trainable_variables
0regularization_losses
1	variables
flayer_metrics
YW
VARIABLE_VALUElayer7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
?

glayers
hlayer_regularization_losses
imetrics
5trainable_variables
jnon_trainable_variables
6regularization_losses
7	variables
klayer_metrics
YW
VARIABLE_VALUElayer8/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer8/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
?

llayers
mlayer_regularization_losses
nmetrics
;trainable_variables
onon_trainable_variables
<regularization_losses
=	variables
player_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
8
0
1
2
3
4
5
6
7
 

q0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	rtotal
	scount
t	variables
u	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

r0
s1

t	variables
|z
VARIABLE_VALUEAdam/layer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer6/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer6/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer7/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer7/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer8/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer8/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer6/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer6/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer7/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer7/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer8/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer8/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_9Placeholder*'
_output_shapes
:?????????/*
dtype0*
shape:?????????/
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_9layer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biaslayer5/kernellayer5/biaslayer6/kernellayer6/biaslayer7/kernellayer7/biaslayer8/kernellayer8/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_331315
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer3/kernel/Read/ReadVariableOplayer3/bias/Read/ReadVariableOp!layer4/kernel/Read/ReadVariableOplayer4/bias/Read/ReadVariableOp!layer5/kernel/Read/ReadVariableOplayer5/bias/Read/ReadVariableOp!layer6/kernel/Read/ReadVariableOplayer6/bias/Read/ReadVariableOp!layer7/kernel/Read/ReadVariableOplayer7/bias/Read/ReadVariableOp!layer8/kernel/Read/ReadVariableOplayer8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/layer1/kernel/m/Read/ReadVariableOp&Adam/layer1/bias/m/Read/ReadVariableOp(Adam/layer2/kernel/m/Read/ReadVariableOp&Adam/layer2/bias/m/Read/ReadVariableOp(Adam/layer3/kernel/m/Read/ReadVariableOp&Adam/layer3/bias/m/Read/ReadVariableOp(Adam/layer4/kernel/m/Read/ReadVariableOp&Adam/layer4/bias/m/Read/ReadVariableOp(Adam/layer5/kernel/m/Read/ReadVariableOp&Adam/layer5/bias/m/Read/ReadVariableOp(Adam/layer6/kernel/m/Read/ReadVariableOp&Adam/layer6/bias/m/Read/ReadVariableOp(Adam/layer7/kernel/m/Read/ReadVariableOp&Adam/layer7/bias/m/Read/ReadVariableOp(Adam/layer8/kernel/m/Read/ReadVariableOp&Adam/layer8/bias/m/Read/ReadVariableOp(Adam/layer1/kernel/v/Read/ReadVariableOp&Adam/layer1/bias/v/Read/ReadVariableOp(Adam/layer2/kernel/v/Read/ReadVariableOp&Adam/layer2/bias/v/Read/ReadVariableOp(Adam/layer3/kernel/v/Read/ReadVariableOp&Adam/layer3/bias/v/Read/ReadVariableOp(Adam/layer4/kernel/v/Read/ReadVariableOp&Adam/layer4/bias/v/Read/ReadVariableOp(Adam/layer5/kernel/v/Read/ReadVariableOp&Adam/layer5/bias/v/Read/ReadVariableOp(Adam/layer6/kernel/v/Read/ReadVariableOp&Adam/layer6/bias/v/Read/ReadVariableOp(Adam/layer7/kernel/v/Read/ReadVariableOp&Adam/layer7/bias/v/Read/ReadVariableOp(Adam/layer8/kernel/v/Read/ReadVariableOp&Adam/layer8/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_332134
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biaslayer5/kernellayer5/biaslayer6/kernellayer6/biaslayer7/kernellayer7/biaslayer8/kernellayer8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/layer1/kernel/mAdam/layer1/bias/mAdam/layer2/kernel/mAdam/layer2/bias/mAdam/layer3/kernel/mAdam/layer3/bias/mAdam/layer4/kernel/mAdam/layer4/bias/mAdam/layer5/kernel/mAdam/layer5/bias/mAdam/layer6/kernel/mAdam/layer6/bias/mAdam/layer7/kernel/mAdam/layer7/bias/mAdam/layer8/kernel/mAdam/layer8/bias/mAdam/layer1/kernel/vAdam/layer1/bias/vAdam/layer2/kernel/vAdam/layer2/bias/vAdam/layer3/kernel/vAdam/layer3/bias/vAdam/layer4/kernel/vAdam/layer4/bias/vAdam/layer5/kernel/vAdam/layer5/bias/vAdam/layer6/kernel/vAdam/layer6/bias/vAdam/layer7/kernel/vAdam/layer7/bias/vAdam/layer8/kernel/vAdam/layer8/bias/v*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_332309??
?
?
B__inference_layer2_layer_call_and_return_conditional_losses_330528

inputs0
matmul_readvariableop_resource:/d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer2/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
,layer2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/d*
dtype02.
,layer2/kernel/Regularizer/Abs/ReadVariableOp?
layer2/kernel/Regularizer/AbsAbs4layer2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/d2
layer2/kernel/Regularizer/Abs?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer2/kernel/Regularizer/Const?
layer2/kernel/Regularizer/SumSum!layer2/kernel/Regularizer/Abs:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/Sum?
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer2/kernel/Regularizer/mul/x?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer2/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer2/kernel/Regularizer/Abs/ReadVariableOp,layer2/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
B__inference_layer2_layer_call_and_return_conditional_losses_331658

inputs0
matmul_readvariableop_resource:/d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer2/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
,layer2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/d*
dtype02.
,layer2/kernel/Regularizer/Abs/ReadVariableOp?
layer2/kernel/Regularizer/AbsAbs4layer2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/d2
layer2/kernel/Regularizer/Abs?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer2/kernel/Regularizer/Const?
layer2/kernel/Regularizer/SumSum!layer2/kernel/Regularizer/Abs:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/Sum?
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer2/kernel/Regularizer/mul/x?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer2/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer2/kernel/Regularizer/Abs/ReadVariableOp,layer2/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
'__inference_layer7_layer_call_fn_331827

inputs
unknown://
	unknown_0:/
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_3306432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
ߔ
?
H__inference_sequential_8_layer_call_and_return_conditional_losses_331529

inputs7
%layer1_matmul_readvariableop_resource://4
&layer1_biasadd_readvariableop_resource:/7
%layer2_matmul_readvariableop_resource:/d4
&layer2_biasadd_readvariableop_resource:d7
%layer3_matmul_readvariableop_resource:dd4
&layer3_biasadd_readvariableop_resource:d7
%layer4_matmul_readvariableop_resource:d/4
&layer4_biasadd_readvariableop_resource:/7
%layer5_matmul_readvariableop_resource://4
&layer5_biasadd_readvariableop_resource:/7
%layer6_matmul_readvariableop_resource://4
&layer6_biasadd_readvariableop_resource:/7
%layer7_matmul_readvariableop_resource://4
&layer7_biasadd_readvariableop_resource:/7
%layer8_matmul_readvariableop_resource:/4
&layer8_biasadd_readvariableop_resource:
identity??layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?,layer1/kernel/Regularizer/Abs/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/MatMul/ReadVariableOp?,layer2/kernel/Regularizer/Abs/ReadVariableOp?layer3/BiasAdd/ReadVariableOp?layer3/MatMul/ReadVariableOp?,layer3/kernel/Regularizer/Abs/ReadVariableOp?layer4/BiasAdd/ReadVariableOp?layer4/MatMul/ReadVariableOp?,layer4/kernel/Regularizer/Abs/ReadVariableOp?layer5/BiasAdd/ReadVariableOp?layer5/MatMul/ReadVariableOp?,layer5/kernel/Regularizer/Abs/ReadVariableOp?layer6/BiasAdd/ReadVariableOp?layer6/MatMul/ReadVariableOp?,layer6/kernel/Regularizer/Abs/ReadVariableOp?layer7/BiasAdd/ReadVariableOp?layer7/MatMul/ReadVariableOp?,layer7/kernel/Regularizer/Abs/ReadVariableOp?layer8/BiasAdd/ReadVariableOp?layer8/MatMul/ReadVariableOp?,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

://*
dtype02
layer1/MatMul/ReadVariableOp?
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer1/MatMul?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer1/BiasAddm
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer1/Relu?
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:/d*
dtype02
layer2/MatMul/ReadVariableOp?
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer2/MatMul?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
layer2/Relu?
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
layer3/MatMul/ReadVariableOp?
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer3/MatMul?
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer3/BiasAdd/ReadVariableOp?
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
layer3/Relu?
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes

:d/*
dtype02
layer4/MatMul/ReadVariableOp?
layer4/MatMulMatMullayer3/Relu:activations:0$layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer4/MatMul?
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer4/BiasAdd/ReadVariableOp?
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer4/BiasAddm
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer4/Relu?
layer5/MatMul/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

://*
dtype02
layer5/MatMul/ReadVariableOp?
layer5/MatMulMatMullayer4/Relu:activations:0$layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer5/MatMul?
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer5/BiasAdd/ReadVariableOp?
layer5/BiasAddBiasAddlayer5/MatMul:product:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer5/BiasAddm
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer5/Relu?
layer6/MatMul/ReadVariableOpReadVariableOp%layer6_matmul_readvariableop_resource*
_output_shapes

://*
dtype02
layer6/MatMul/ReadVariableOp?
layer6/MatMulMatMullayer5/Relu:activations:0$layer6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer6/MatMul?
layer6/BiasAdd/ReadVariableOpReadVariableOp&layer6_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer6/BiasAdd/ReadVariableOp?
layer6/BiasAddBiasAddlayer6/MatMul:product:0%layer6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer6/BiasAddm
layer6/ReluRelulayer6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer6/Relu?
layer7/MatMul/ReadVariableOpReadVariableOp%layer7_matmul_readvariableop_resource*
_output_shapes

://*
dtype02
layer7/MatMul/ReadVariableOp?
layer7/MatMulMatMullayer6/Relu:activations:0$layer7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer7/MatMul?
layer7/BiasAdd/ReadVariableOpReadVariableOp&layer7_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer7/BiasAdd/ReadVariableOp?
layer7/BiasAddBiasAddlayer7/MatMul:product:0%layer7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer7/BiasAddm
layer7/ReluRelulayer7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer7/Relu?
layer8/MatMul/ReadVariableOpReadVariableOp%layer8_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02
layer8/MatMul/ReadVariableOp?
layer8/MatMulMatMullayer7/Relu:activations:0$layer8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer8/MatMul?
layer8/BiasAdd/ReadVariableOpReadVariableOp&layer8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer8/BiasAdd/ReadVariableOp?
layer8/BiasAddBiasAddlayer8/MatMul:product:0%layer8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer8/BiasAdd?
,layer1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer1/kernel/Regularizer/Abs/ReadVariableOp?
layer1/kernel/Regularizer/AbsAbs4layer1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer1/kernel/Regularizer/Abs?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer1/kernel/Regularizer/Const?
layer1/kernel/Regularizer/SumSum!layer1/kernel/Regularizer/Abs:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/Sum?
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer1/kernel/Regularizer/mul/x?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/mul?
,layer2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:/d*
dtype02.
,layer2/kernel/Regularizer/Abs/ReadVariableOp?
layer2/kernel/Regularizer/AbsAbs4layer2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/d2
layer2/kernel/Regularizer/Abs?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer2/kernel/Regularizer/Const?
layer2/kernel/Regularizer/SumSum!layer2/kernel/Regularizer/Abs:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/Sum?
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer2/kernel/Regularizer/mul/x?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/mul?
,layer3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,layer3/kernel/Regularizer/Abs/ReadVariableOp?
layer3/kernel/Regularizer/AbsAbs4layer3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2
layer3/kernel/Regularizer/Abs?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer3/kernel/Regularizer/Const?
layer3/kernel/Regularizer/SumSum!layer3/kernel/Regularizer/Abs:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/Sum?
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer3/kernel/Regularizer/mul/x?
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/mul?
,layer4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes

:d/*
dtype02.
,layer4/kernel/Regularizer/Abs/ReadVariableOp?
layer4/kernel/Regularizer/AbsAbs4layer4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d/2
layer4/kernel/Regularizer/Abs?
layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer4/kernel/Regularizer/Const?
layer4/kernel/Regularizer/SumSum!layer4/kernel/Regularizer/Abs:y:0(layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/Sum?
layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer4/kernel/Regularizer/mul/x?
layer4/kernel/Regularizer/mulMul(layer4/kernel/Regularizer/mul/x:output:0&layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/mul?
,layer5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer5/kernel/Regularizer/Abs/ReadVariableOp?
layer5/kernel/Regularizer/AbsAbs4layer5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer5/kernel/Regularizer/Abs?
layer5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer5/kernel/Regularizer/Const?
layer5/kernel/Regularizer/SumSum!layer5/kernel/Regularizer/Abs:y:0(layer5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/Sum?
layer5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer5/kernel/Regularizer/mul/x?
layer5/kernel/Regularizer/mulMul(layer5/kernel/Regularizer/mul/x:output:0&layer5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/mul?
,layer6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer6_matmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer6/kernel/Regularizer/Abs/ReadVariableOp?
layer6/kernel/Regularizer/AbsAbs4layer6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer6/kernel/Regularizer/Abs?
layer6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer6/kernel/Regularizer/Const?
layer6/kernel/Regularizer/SumSum!layer6/kernel/Regularizer/Abs:y:0(layer6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/Sum?
layer6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer6/kernel/Regularizer/mul/x?
layer6/kernel/Regularizer/mulMul(layer6/kernel/Regularizer/mul/x:output:0&layer6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/mul?
,layer7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer7_matmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer7/kernel/Regularizer/Abs/ReadVariableOp?
layer7/kernel/Regularizer/AbsAbs4layer7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer7/kernel/Regularizer/Abs?
layer7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer7/kernel/Regularizer/Const?
layer7/kernel/Regularizer/SumSum!layer7/kernel/Regularizer/Abs:y:0(layer7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/Sum?
layer7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer7/kernel/Regularizer/mul/x?
layer7/kernel/Regularizer/mulMul(layer7/kernel/Regularizer/mul/x:output:0&layer7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/mul?
,layer8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer8_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02.
,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer8/kernel/Regularizer/AbsAbs4layer8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/2
layer8/kernel/Regularizer/Abs?
layer8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer8/kernel/Regularizer/Const?
layer8/kernel/Regularizer/SumSum!layer8/kernel/Regularizer/Abs:y:0(layer8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/Sum?
layer8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer8/kernel/Regularizer/mul/x?
layer8/kernel/Regularizer/mulMul(layer8/kernel/Regularizer/mul/x:output:0&layer8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/mulr
IdentityIdentitylayer8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp-^layer1/kernel/Regularizer/Abs/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp-^layer2/kernel/Regularizer/Abs/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp-^layer3/kernel/Regularizer/Abs/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp-^layer4/kernel/Regularizer/Abs/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/MatMul/ReadVariableOp-^layer5/kernel/Regularizer/Abs/ReadVariableOp^layer6/BiasAdd/ReadVariableOp^layer6/MatMul/ReadVariableOp-^layer6/kernel/Regularizer/Abs/ReadVariableOp^layer7/BiasAdd/ReadVariableOp^layer7/MatMul/ReadVariableOp-^layer7/kernel/Regularizer/Abs/ReadVariableOp^layer8/BiasAdd/ReadVariableOp^layer8/MatMul/ReadVariableOp-^layer8/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2\
,layer1/kernel/Regularizer/Abs/ReadVariableOp,layer1/kernel/Regularizer/Abs/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2\
,layer2/kernel/Regularizer/Abs/ReadVariableOp,layer2/kernel/Regularizer/Abs/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2\
,layer3/kernel/Regularizer/Abs/ReadVariableOp,layer3/kernel/Regularizer/Abs/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2\
,layer4/kernel/Regularizer/Abs/ReadVariableOp,layer4/kernel/Regularizer/Abs/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/MatMul/ReadVariableOplayer5/MatMul/ReadVariableOp2\
,layer5/kernel/Regularizer/Abs/ReadVariableOp,layer5/kernel/Regularizer/Abs/ReadVariableOp2>
layer6/BiasAdd/ReadVariableOplayer6/BiasAdd/ReadVariableOp2<
layer6/MatMul/ReadVariableOplayer6/MatMul/ReadVariableOp2\
,layer6/kernel/Regularizer/Abs/ReadVariableOp,layer6/kernel/Regularizer/Abs/ReadVariableOp2>
layer7/BiasAdd/ReadVariableOplayer7/BiasAdd/ReadVariableOp2<
layer7/MatMul/ReadVariableOplayer7/MatMul/ReadVariableOp2\
,layer7/kernel/Regularizer/Abs/ReadVariableOp,layer7/kernel/Regularizer/Abs/ReadVariableOp2>
layer8/BiasAdd/ReadVariableOplayer8/BiasAdd/ReadVariableOp2<
layer8/MatMul/ReadVariableOplayer8/MatMul/ReadVariableOp2\
,layer8/kernel/Regularizer/Abs/ReadVariableOp,layer8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
B__inference_layer1_layer_call_and_return_conditional_losses_331626

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer1/kernel/Regularizer/Abs/ReadVariableOp?
layer1/kernel/Regularizer/AbsAbs4layer1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer1/kernel/Regularizer/Abs?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer1/kernel/Regularizer/Const?
layer1/kernel/Regularizer/SumSum!layer1/kernel/Regularizer/Abs:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/Sum?
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer1/kernel/Regularizer/mul/x?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer1/kernel/Regularizer/Abs/ReadVariableOp,layer1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
B__inference_layer7_layer_call_and_return_conditional_losses_331818

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer7/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer7/kernel/Regularizer/Abs/ReadVariableOp?
layer7/kernel/Regularizer/AbsAbs4layer7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer7/kernel/Regularizer/Abs?
layer7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer7/kernel/Regularizer/Const?
layer7/kernel/Regularizer/SumSum!layer7/kernel/Regularizer/Abs:y:0(layer7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/Sum?
layer7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer7/kernel/Regularizer/mul/x?
layer7/kernel/Regularizer/mulMul(layer7/kernel/Regularizer/mul/x:output:0&layer7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer7/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer7/kernel/Regularizer/Abs/ReadVariableOp,layer7/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
-__inference_sequential_8_layer_call_fn_331603

inputs
unknown://
	unknown_0:/
	unknown_1:/d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d/
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_3309662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?u
?	
H__inference_sequential_8_layer_call_and_return_conditional_losses_331222
input_9
layer1_331133://
layer1_331135:/
layer2_331138:/d
layer2_331140:d
layer3_331143:dd
layer3_331145:d
layer4_331148:d/
layer4_331150:/
layer5_331153://
layer5_331155:/
layer6_331158://
layer6_331160:/
layer7_331163://
layer7_331165:/
layer8_331168:/
layer8_331170:
identity??layer1/StatefulPartitionedCall?,layer1/kernel/Regularizer/Abs/ReadVariableOp?layer2/StatefulPartitionedCall?,layer2/kernel/Regularizer/Abs/ReadVariableOp?layer3/StatefulPartitionedCall?,layer3/kernel/Regularizer/Abs/ReadVariableOp?layer4/StatefulPartitionedCall?,layer4/kernel/Regularizer/Abs/ReadVariableOp?layer5/StatefulPartitionedCall?,layer5/kernel/Regularizer/Abs/ReadVariableOp?layer6/StatefulPartitionedCall?,layer6/kernel/Regularizer/Abs/ReadVariableOp?layer7/StatefulPartitionedCall?,layer7/kernel/Regularizer/Abs/ReadVariableOp?layer8/StatefulPartitionedCall?,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_9layer1_331133layer1_331135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_3305052 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_331138layer2_331140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_3305282 
layer2/StatefulPartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_331143layer3_331145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_3305512 
layer3/StatefulPartitionedCall?
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_331148layer4_331150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_3305742 
layer4/StatefulPartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_331153layer5_331155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_3305972 
layer5/StatefulPartitionedCall?
layer6/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0layer6_331158layer6_331160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_3306202 
layer6/StatefulPartitionedCall?
layer7/StatefulPartitionedCallStatefulPartitionedCall'layer6/StatefulPartitionedCall:output:0layer7_331163layer7_331165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_3306432 
layer7/StatefulPartitionedCall?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_331168layer8_331170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_3306652 
layer8/StatefulPartitionedCall?
,layer1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer1_331133*
_output_shapes

://*
dtype02.
,layer1/kernel/Regularizer/Abs/ReadVariableOp?
layer1/kernel/Regularizer/AbsAbs4layer1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer1/kernel/Regularizer/Abs?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer1/kernel/Regularizer/Const?
layer1/kernel/Regularizer/SumSum!layer1/kernel/Regularizer/Abs:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/Sum?
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer1/kernel/Regularizer/mul/x?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/mul?
,layer2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer2_331138*
_output_shapes

:/d*
dtype02.
,layer2/kernel/Regularizer/Abs/ReadVariableOp?
layer2/kernel/Regularizer/AbsAbs4layer2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/d2
layer2/kernel/Regularizer/Abs?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer2/kernel/Regularizer/Const?
layer2/kernel/Regularizer/SumSum!layer2/kernel/Regularizer/Abs:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/Sum?
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer2/kernel/Regularizer/mul/x?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/mul?
,layer3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer3_331143*
_output_shapes

:dd*
dtype02.
,layer3/kernel/Regularizer/Abs/ReadVariableOp?
layer3/kernel/Regularizer/AbsAbs4layer3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2
layer3/kernel/Regularizer/Abs?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer3/kernel/Regularizer/Const?
layer3/kernel/Regularizer/SumSum!layer3/kernel/Regularizer/Abs:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/Sum?
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer3/kernel/Regularizer/mul/x?
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/mul?
,layer4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer4_331148*
_output_shapes

:d/*
dtype02.
,layer4/kernel/Regularizer/Abs/ReadVariableOp?
layer4/kernel/Regularizer/AbsAbs4layer4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d/2
layer4/kernel/Regularizer/Abs?
layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer4/kernel/Regularizer/Const?
layer4/kernel/Regularizer/SumSum!layer4/kernel/Regularizer/Abs:y:0(layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/Sum?
layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer4/kernel/Regularizer/mul/x?
layer4/kernel/Regularizer/mulMul(layer4/kernel/Regularizer/mul/x:output:0&layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/mul?
,layer5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer5_331153*
_output_shapes

://*
dtype02.
,layer5/kernel/Regularizer/Abs/ReadVariableOp?
layer5/kernel/Regularizer/AbsAbs4layer5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer5/kernel/Regularizer/Abs?
layer5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer5/kernel/Regularizer/Const?
layer5/kernel/Regularizer/SumSum!layer5/kernel/Regularizer/Abs:y:0(layer5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/Sum?
layer5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer5/kernel/Regularizer/mul/x?
layer5/kernel/Regularizer/mulMul(layer5/kernel/Regularizer/mul/x:output:0&layer5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/mul?
,layer6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer6_331158*
_output_shapes

://*
dtype02.
,layer6/kernel/Regularizer/Abs/ReadVariableOp?
layer6/kernel/Regularizer/AbsAbs4layer6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer6/kernel/Regularizer/Abs?
layer6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer6/kernel/Regularizer/Const?
layer6/kernel/Regularizer/SumSum!layer6/kernel/Regularizer/Abs:y:0(layer6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/Sum?
layer6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer6/kernel/Regularizer/mul/x?
layer6/kernel/Regularizer/mulMul(layer6/kernel/Regularizer/mul/x:output:0&layer6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/mul?
,layer7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer7_331163*
_output_shapes

://*
dtype02.
,layer7/kernel/Regularizer/Abs/ReadVariableOp?
layer7/kernel/Regularizer/AbsAbs4layer7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer7/kernel/Regularizer/Abs?
layer7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer7/kernel/Regularizer/Const?
layer7/kernel/Regularizer/SumSum!layer7/kernel/Regularizer/Abs:y:0(layer7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/Sum?
layer7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer7/kernel/Regularizer/mul/x?
layer7/kernel/Regularizer/mulMul(layer7/kernel/Regularizer/mul/x:output:0&layer7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/mul?
,layer8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer8_331168*
_output_shapes

:/*
dtype02.
,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer8/kernel/Regularizer/AbsAbs4layer8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/2
layer8/kernel/Regularizer/Abs?
layer8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer8/kernel/Regularizer/Const?
layer8/kernel/Regularizer/SumSum!layer8/kernel/Regularizer/Abs:y:0(layer8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/Sum?
layer8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer8/kernel/Regularizer/mul/x?
layer8/kernel/Regularizer/mulMul(layer8/kernel/Regularizer/mul/x:output:0&layer8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/mul?
IdentityIdentity'layer8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^layer1/StatefulPartitionedCall-^layer1/kernel/Regularizer/Abs/ReadVariableOp^layer2/StatefulPartitionedCall-^layer2/kernel/Regularizer/Abs/ReadVariableOp^layer3/StatefulPartitionedCall-^layer3/kernel/Regularizer/Abs/ReadVariableOp^layer4/StatefulPartitionedCall-^layer4/kernel/Regularizer/Abs/ReadVariableOp^layer5/StatefulPartitionedCall-^layer5/kernel/Regularizer/Abs/ReadVariableOp^layer6/StatefulPartitionedCall-^layer6/kernel/Regularizer/Abs/ReadVariableOp^layer7/StatefulPartitionedCall-^layer7/kernel/Regularizer/Abs/ReadVariableOp^layer8/StatefulPartitionedCall-^layer8/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2\
,layer1/kernel/Regularizer/Abs/ReadVariableOp,layer1/kernel/Regularizer/Abs/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2\
,layer2/kernel/Regularizer/Abs/ReadVariableOp,layer2/kernel/Regularizer/Abs/ReadVariableOp2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2\
,layer3/kernel/Regularizer/Abs/ReadVariableOp,layer3/kernel/Regularizer/Abs/ReadVariableOp2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2\
,layer4/kernel/Regularizer/Abs/ReadVariableOp,layer4/kernel/Regularizer/Abs/ReadVariableOp2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2\
,layer5/kernel/Regularizer/Abs/ReadVariableOp,layer5/kernel/Regularizer/Abs/ReadVariableOp2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall2\
,layer6/kernel/Regularizer/Abs/ReadVariableOp,layer6/kernel/Regularizer/Abs/ReadVariableOp2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2\
,layer7/kernel/Regularizer/Abs/ReadVariableOp,layer7/kernel/Regularizer/Abs/ReadVariableOp2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2\
,layer8/kernel/Regularizer/Abs/ReadVariableOp,layer8/kernel/Regularizer/Abs/ReadVariableOp:P L
'
_output_shapes
:?????????/
!
_user_specified_name	input_9
?
?
B__inference_layer7_layer_call_and_return_conditional_losses_330643

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer7/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer7/kernel/Regularizer/Abs/ReadVariableOp?
layer7/kernel/Regularizer/AbsAbs4layer7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer7/kernel/Regularizer/Abs?
layer7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer7/kernel/Regularizer/Const?
layer7/kernel/Regularizer/SumSum!layer7/kernel/Regularizer/Abs:y:0(layer7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/Sum?
layer7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer7/kernel/Regularizer/mul/x?
layer7/kernel/Regularizer/mulMul(layer7/kernel/Regularizer/mul/x:output:0&layer7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer7/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer7/kernel/Regularizer/Abs/ReadVariableOp,layer7/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
B__inference_layer4_layer_call_and_return_conditional_losses_331722

inputs0
matmul_readvariableop_resource:d/-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer4/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d/*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d/*
dtype02.
,layer4/kernel/Regularizer/Abs/ReadVariableOp?
layer4/kernel/Regularizer/AbsAbs4layer4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d/2
layer4/kernel/Regularizer/Abs?
layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer4/kernel/Regularizer/Const?
layer4/kernel/Regularizer/SumSum!layer4/kernel/Regularizer/Abs:y:0(layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/Sum?
layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer4/kernel/Regularizer/mul/x?
layer4/kernel/Regularizer/mulMul(layer4/kernel/Regularizer/mul/x:output:0&layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer4/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer4/kernel/Regularizer/Abs/ReadVariableOp,layer4/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
'__inference_layer8_layer_call_fn_331858

inputs
unknown:/
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_3306652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
__inference_loss_fn_6_331935G
5layer7_kernel_regularizer_abs_readvariableop_resource://
identity??,layer7/kernel/Regularizer/Abs/ReadVariableOp?
,layer7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5layer7_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer7/kernel/Regularizer/Abs/ReadVariableOp?
layer7/kernel/Regularizer/AbsAbs4layer7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer7/kernel/Regularizer/Abs?
layer7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer7/kernel/Regularizer/Const?
layer7/kernel/Regularizer/SumSum!layer7/kernel/Regularizer/Abs:y:0(layer7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/Sum?
layer7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer7/kernel/Regularizer/mul/x?
layer7/kernel/Regularizer/mulMul(layer7/kernel/Regularizer/mul/x:output:0&layer7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/mulk
IdentityIdentity!layer7/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^layer7/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,layer7/kernel/Regularizer/Abs/ReadVariableOp,layer7/kernel/Regularizer/Abs/ReadVariableOp
??
? 
"__inference__traced_restore_332309
file_prefix0
assignvariableop_layer1_kernel://,
assignvariableop_1_layer1_bias:/2
 assignvariableop_2_layer2_kernel:/d,
assignvariableop_3_layer2_bias:d2
 assignvariableop_4_layer3_kernel:dd,
assignvariableop_5_layer3_bias:d2
 assignvariableop_6_layer4_kernel:d/,
assignvariableop_7_layer4_bias:/2
 assignvariableop_8_layer5_kernel://,
assignvariableop_9_layer5_bias:/3
!assignvariableop_10_layer6_kernel://-
assignvariableop_11_layer6_bias:/3
!assignvariableop_12_layer7_kernel://-
assignvariableop_13_layer7_bias:/3
!assignvariableop_14_layer8_kernel:/-
assignvariableop_15_layer8_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: #
assignvariableop_22_count: :
(assignvariableop_23_adam_layer1_kernel_m://4
&assignvariableop_24_adam_layer1_bias_m:/:
(assignvariableop_25_adam_layer2_kernel_m:/d4
&assignvariableop_26_adam_layer2_bias_m:d:
(assignvariableop_27_adam_layer3_kernel_m:dd4
&assignvariableop_28_adam_layer3_bias_m:d:
(assignvariableop_29_adam_layer4_kernel_m:d/4
&assignvariableop_30_adam_layer4_bias_m:/:
(assignvariableop_31_adam_layer5_kernel_m://4
&assignvariableop_32_adam_layer5_bias_m:/:
(assignvariableop_33_adam_layer6_kernel_m://4
&assignvariableop_34_adam_layer6_bias_m:/:
(assignvariableop_35_adam_layer7_kernel_m://4
&assignvariableop_36_adam_layer7_bias_m:/:
(assignvariableop_37_adam_layer8_kernel_m:/4
&assignvariableop_38_adam_layer8_bias_m::
(assignvariableop_39_adam_layer1_kernel_v://4
&assignvariableop_40_adam_layer1_bias_v:/:
(assignvariableop_41_adam_layer2_kernel_v:/d4
&assignvariableop_42_adam_layer2_bias_v:d:
(assignvariableop_43_adam_layer3_kernel_v:dd4
&assignvariableop_44_adam_layer3_bias_v:d:
(assignvariableop_45_adam_layer4_kernel_v:d/4
&assignvariableop_46_adam_layer4_bias_v:/:
(assignvariableop_47_adam_layer5_kernel_v://4
&assignvariableop_48_adam_layer5_bias_v:/:
(assignvariableop_49_adam_layer6_kernel_v://4
&assignvariableop_50_adam_layer6_bias_v:/:
(assignvariableop_51_adam_layer7_kernel_v://4
&assignvariableop_52_adam_layer7_bias_v:/:
(assignvariableop_53_adam_layer8_kernel_v:/4
&assignvariableop_54_adam_layer8_bias_v:
identity_56??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_layer6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_layer6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_layer7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_layer7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_layer8_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_layer8_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_layer1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_layer1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_layer2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_layer2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_layer3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_layer3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_layer4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_layer4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_layer5_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_layer5_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_layer6_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_layer6_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_layer7_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_layer7_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_layer8_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_layer8_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_layer1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_layer1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_layer2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_layer2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_layer3_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_layer3_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_layer4_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_layer4_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_layer5_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_layer5_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_layer6_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_layer6_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_layer7_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_layer7_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_layer8_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_layer8_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55f
Identity_56IdentityIdentity_55:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_56?

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_56Identity_56:output:0*?
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
'__inference_layer4_layer_call_fn_331731

inputs
unknown:d/
	unknown_0:/
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_3305742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?m
?
__inference__traced_save_332134
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer3_kernel_read_readvariableop*
&savev2_layer3_bias_read_readvariableop,
(savev2_layer4_kernel_read_readvariableop*
&savev2_layer4_bias_read_readvariableop,
(savev2_layer5_kernel_read_readvariableop*
&savev2_layer5_bias_read_readvariableop,
(savev2_layer6_kernel_read_readvariableop*
&savev2_layer6_bias_read_readvariableop,
(savev2_layer7_kernel_read_readvariableop*
&savev2_layer7_bias_read_readvariableop,
(savev2_layer8_kernel_read_readvariableop*
&savev2_layer8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_layer1_kernel_m_read_readvariableop1
-savev2_adam_layer1_bias_m_read_readvariableop3
/savev2_adam_layer2_kernel_m_read_readvariableop1
-savev2_adam_layer2_bias_m_read_readvariableop3
/savev2_adam_layer3_kernel_m_read_readvariableop1
-savev2_adam_layer3_bias_m_read_readvariableop3
/savev2_adam_layer4_kernel_m_read_readvariableop1
-savev2_adam_layer4_bias_m_read_readvariableop3
/savev2_adam_layer5_kernel_m_read_readvariableop1
-savev2_adam_layer5_bias_m_read_readvariableop3
/savev2_adam_layer6_kernel_m_read_readvariableop1
-savev2_adam_layer6_bias_m_read_readvariableop3
/savev2_adam_layer7_kernel_m_read_readvariableop1
-savev2_adam_layer7_bias_m_read_readvariableop3
/savev2_adam_layer8_kernel_m_read_readvariableop1
-savev2_adam_layer8_bias_m_read_readvariableop3
/savev2_adam_layer1_kernel_v_read_readvariableop1
-savev2_adam_layer1_bias_v_read_readvariableop3
/savev2_adam_layer2_kernel_v_read_readvariableop1
-savev2_adam_layer2_bias_v_read_readvariableop3
/savev2_adam_layer3_kernel_v_read_readvariableop1
-savev2_adam_layer3_bias_v_read_readvariableop3
/savev2_adam_layer4_kernel_v_read_readvariableop1
-savev2_adam_layer4_bias_v_read_readvariableop3
/savev2_adam_layer5_kernel_v_read_readvariableop1
-savev2_adam_layer5_bias_v_read_readvariableop3
/savev2_adam_layer6_kernel_v_read_readvariableop1
-savev2_adam_layer6_bias_v_read_readvariableop3
/savev2_adam_layer7_kernel_v_read_readvariableop1
-savev2_adam_layer7_bias_v_read_readvariableop3
/savev2_adam_layer8_kernel_v_read_readvariableop1
-savev2_adam_layer8_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop(savev2_layer4_kernel_read_readvariableop&savev2_layer4_bias_read_readvariableop(savev2_layer5_kernel_read_readvariableop&savev2_layer5_bias_read_readvariableop(savev2_layer6_kernel_read_readvariableop&savev2_layer6_bias_read_readvariableop(savev2_layer7_kernel_read_readvariableop&savev2_layer7_bias_read_readvariableop(savev2_layer8_kernel_read_readvariableop&savev2_layer8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop/savev2_adam_layer3_kernel_m_read_readvariableop-savev2_adam_layer3_bias_m_read_readvariableop/savev2_adam_layer4_kernel_m_read_readvariableop-savev2_adam_layer4_bias_m_read_readvariableop/savev2_adam_layer5_kernel_m_read_readvariableop-savev2_adam_layer5_bias_m_read_readvariableop/savev2_adam_layer6_kernel_m_read_readvariableop-savev2_adam_layer6_bias_m_read_readvariableop/savev2_adam_layer7_kernel_m_read_readvariableop-savev2_adam_layer7_bias_m_read_readvariableop/savev2_adam_layer8_kernel_m_read_readvariableop-savev2_adam_layer8_bias_m_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop/savev2_adam_layer3_kernel_v_read_readvariableop-savev2_adam_layer3_bias_v_read_readvariableop/savev2_adam_layer4_kernel_v_read_readvariableop-savev2_adam_layer4_bias_v_read_readvariableop/savev2_adam_layer5_kernel_v_read_readvariableop-savev2_adam_layer5_bias_v_read_readvariableop/savev2_adam_layer6_kernel_v_read_readvariableop-savev2_adam_layer6_bias_v_read_readvariableop/savev2_adam_layer7_kernel_v_read_readvariableop-savev2_adam_layer7_bias_v_read_readvariableop/savev2_adam_layer8_kernel_v_read_readvariableop-savev2_adam_layer8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ://:/:/d:d:dd:d:d/:/://:/://:/://:/:/:: : : : : : : ://:/:/d:d:dd:d:d/:/://:/://:/://:/:/:://:/:/d:d:dd:d:d/:/://:/://:/://:/:/:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d/: 

_output_shapes
:/:$	 

_output_shapes

://: 


_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d/: 

_output_shapes
:/:$  

_output_shapes

://: !

_output_shapes
:/:$" 

_output_shapes

://: #

_output_shapes
:/:$$ 

_output_shapes

://: %

_output_shapes
:/:$& 

_output_shapes

:/: '

_output_shapes
::$( 

_output_shapes

://: )

_output_shapes
:/:$* 

_output_shapes

:/d: +

_output_shapes
:d:$, 

_output_shapes

:dd: -

_output_shapes
:d:$. 

_output_shapes

:d/: /

_output_shapes
:/:$0 

_output_shapes

://: 1

_output_shapes
:/:$2 

_output_shapes

://: 3

_output_shapes
:/:$4 

_output_shapes

://: 5

_output_shapes
:/:$6 

_output_shapes

:/: 7

_output_shapes
::8

_output_shapes
: 
?
?
__inference_loss_fn_4_331913G
5layer5_kernel_regularizer_abs_readvariableop_resource://
identity??,layer5/kernel/Regularizer/Abs/ReadVariableOp?
,layer5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5layer5_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer5/kernel/Regularizer/Abs/ReadVariableOp?
layer5/kernel/Regularizer/AbsAbs4layer5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer5/kernel/Regularizer/Abs?
layer5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer5/kernel/Regularizer/Const?
layer5/kernel/Regularizer/SumSum!layer5/kernel/Regularizer/Abs:y:0(layer5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/Sum?
layer5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer5/kernel/Regularizer/mul/x?
layer5/kernel/Regularizer/mulMul(layer5/kernel/Regularizer/mul/x:output:0&layer5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/mulk
IdentityIdentity!layer5/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^layer5/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,layer5/kernel/Regularizer/Abs/ReadVariableOp,layer5/kernel/Regularizer/Abs/ReadVariableOp
?
?
B__inference_layer8_layer_call_and_return_conditional_losses_330665

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer8/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
,layer8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype02.
,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer8/kernel/Regularizer/AbsAbs4layer8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/2
layer8/kernel/Regularizer/Abs?
layer8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer8/kernel/Regularizer/Const?
layer8/kernel/Regularizer/SumSum!layer8/kernel/Regularizer/Abs:y:0(layer8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/Sum?
layer8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer8/kernel/Regularizer/mul/x?
layer8/kernel/Regularizer/mulMul(layer8/kernel/Regularizer/mul/x:output:0&layer8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer8/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer8/kernel/Regularizer/Abs/ReadVariableOp,layer8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
B__inference_layer3_layer_call_and_return_conditional_losses_330551

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer3/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
,layer3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,layer3/kernel/Regularizer/Abs/ReadVariableOp?
layer3/kernel/Regularizer/AbsAbs4layer3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2
layer3/kernel/Regularizer/Abs?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer3/kernel/Regularizer/Const?
layer3/kernel/Regularizer/SumSum!layer3/kernel/Regularizer/Abs:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/Sum?
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer3/kernel/Regularizer/mul/x?
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer3/kernel/Regularizer/Abs/ReadVariableOp,layer3/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
B__inference_layer6_layer_call_and_return_conditional_losses_331786

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer6/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer6/kernel/Regularizer/Abs/ReadVariableOp?
layer6/kernel/Regularizer/AbsAbs4layer6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer6/kernel/Regularizer/Abs?
layer6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer6/kernel/Regularizer/Const?
layer6/kernel/Regularizer/SumSum!layer6/kernel/Regularizer/Abs:y:0(layer6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/Sum?
layer6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer6/kernel/Regularizer/mul/x?
layer6/kernel/Regularizer/mulMul(layer6/kernel/Regularizer/mul/x:output:0&layer6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer6/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer6/kernel/Regularizer/Abs/ReadVariableOp,layer6/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?u
?	
H__inference_sequential_8_layer_call_and_return_conditional_losses_331130
input_9
layer1_331041://
layer1_331043:/
layer2_331046:/d
layer2_331048:d
layer3_331051:dd
layer3_331053:d
layer4_331056:d/
layer4_331058:/
layer5_331061://
layer5_331063:/
layer6_331066://
layer6_331068:/
layer7_331071://
layer7_331073:/
layer8_331076:/
layer8_331078:
identity??layer1/StatefulPartitionedCall?,layer1/kernel/Regularizer/Abs/ReadVariableOp?layer2/StatefulPartitionedCall?,layer2/kernel/Regularizer/Abs/ReadVariableOp?layer3/StatefulPartitionedCall?,layer3/kernel/Regularizer/Abs/ReadVariableOp?layer4/StatefulPartitionedCall?,layer4/kernel/Regularizer/Abs/ReadVariableOp?layer5/StatefulPartitionedCall?,layer5/kernel/Regularizer/Abs/ReadVariableOp?layer6/StatefulPartitionedCall?,layer6/kernel/Regularizer/Abs/ReadVariableOp?layer7/StatefulPartitionedCall?,layer7/kernel/Regularizer/Abs/ReadVariableOp?layer8/StatefulPartitionedCall?,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_9layer1_331041layer1_331043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_3305052 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_331046layer2_331048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_3305282 
layer2/StatefulPartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_331051layer3_331053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_3305512 
layer3/StatefulPartitionedCall?
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_331056layer4_331058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_3305742 
layer4/StatefulPartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_331061layer5_331063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_3305972 
layer5/StatefulPartitionedCall?
layer6/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0layer6_331066layer6_331068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_3306202 
layer6/StatefulPartitionedCall?
layer7/StatefulPartitionedCallStatefulPartitionedCall'layer6/StatefulPartitionedCall:output:0layer7_331071layer7_331073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_3306432 
layer7/StatefulPartitionedCall?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_331076layer8_331078*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_3306652 
layer8/StatefulPartitionedCall?
,layer1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer1_331041*
_output_shapes

://*
dtype02.
,layer1/kernel/Regularizer/Abs/ReadVariableOp?
layer1/kernel/Regularizer/AbsAbs4layer1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer1/kernel/Regularizer/Abs?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer1/kernel/Regularizer/Const?
layer1/kernel/Regularizer/SumSum!layer1/kernel/Regularizer/Abs:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/Sum?
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer1/kernel/Regularizer/mul/x?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/mul?
,layer2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer2_331046*
_output_shapes

:/d*
dtype02.
,layer2/kernel/Regularizer/Abs/ReadVariableOp?
layer2/kernel/Regularizer/AbsAbs4layer2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/d2
layer2/kernel/Regularizer/Abs?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer2/kernel/Regularizer/Const?
layer2/kernel/Regularizer/SumSum!layer2/kernel/Regularizer/Abs:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/Sum?
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer2/kernel/Regularizer/mul/x?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/mul?
,layer3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer3_331051*
_output_shapes

:dd*
dtype02.
,layer3/kernel/Regularizer/Abs/ReadVariableOp?
layer3/kernel/Regularizer/AbsAbs4layer3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2
layer3/kernel/Regularizer/Abs?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer3/kernel/Regularizer/Const?
layer3/kernel/Regularizer/SumSum!layer3/kernel/Regularizer/Abs:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/Sum?
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer3/kernel/Regularizer/mul/x?
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/mul?
,layer4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer4_331056*
_output_shapes

:d/*
dtype02.
,layer4/kernel/Regularizer/Abs/ReadVariableOp?
layer4/kernel/Regularizer/AbsAbs4layer4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d/2
layer4/kernel/Regularizer/Abs?
layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer4/kernel/Regularizer/Const?
layer4/kernel/Regularizer/SumSum!layer4/kernel/Regularizer/Abs:y:0(layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/Sum?
layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer4/kernel/Regularizer/mul/x?
layer4/kernel/Regularizer/mulMul(layer4/kernel/Regularizer/mul/x:output:0&layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/mul?
,layer5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer5_331061*
_output_shapes

://*
dtype02.
,layer5/kernel/Regularizer/Abs/ReadVariableOp?
layer5/kernel/Regularizer/AbsAbs4layer5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer5/kernel/Regularizer/Abs?
layer5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer5/kernel/Regularizer/Const?
layer5/kernel/Regularizer/SumSum!layer5/kernel/Regularizer/Abs:y:0(layer5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/Sum?
layer5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer5/kernel/Regularizer/mul/x?
layer5/kernel/Regularizer/mulMul(layer5/kernel/Regularizer/mul/x:output:0&layer5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/mul?
,layer6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer6_331066*
_output_shapes

://*
dtype02.
,layer6/kernel/Regularizer/Abs/ReadVariableOp?
layer6/kernel/Regularizer/AbsAbs4layer6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer6/kernel/Regularizer/Abs?
layer6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer6/kernel/Regularizer/Const?
layer6/kernel/Regularizer/SumSum!layer6/kernel/Regularizer/Abs:y:0(layer6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/Sum?
layer6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer6/kernel/Regularizer/mul/x?
layer6/kernel/Regularizer/mulMul(layer6/kernel/Regularizer/mul/x:output:0&layer6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/mul?
,layer7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer7_331071*
_output_shapes

://*
dtype02.
,layer7/kernel/Regularizer/Abs/ReadVariableOp?
layer7/kernel/Regularizer/AbsAbs4layer7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer7/kernel/Regularizer/Abs?
layer7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer7/kernel/Regularizer/Const?
layer7/kernel/Regularizer/SumSum!layer7/kernel/Regularizer/Abs:y:0(layer7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/Sum?
layer7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer7/kernel/Regularizer/mul/x?
layer7/kernel/Regularizer/mulMul(layer7/kernel/Regularizer/mul/x:output:0&layer7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/mul?
,layer8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer8_331076*
_output_shapes

:/*
dtype02.
,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer8/kernel/Regularizer/AbsAbs4layer8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/2
layer8/kernel/Regularizer/Abs?
layer8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer8/kernel/Regularizer/Const?
layer8/kernel/Regularizer/SumSum!layer8/kernel/Regularizer/Abs:y:0(layer8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/Sum?
layer8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer8/kernel/Regularizer/mul/x?
layer8/kernel/Regularizer/mulMul(layer8/kernel/Regularizer/mul/x:output:0&layer8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/mul?
IdentityIdentity'layer8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^layer1/StatefulPartitionedCall-^layer1/kernel/Regularizer/Abs/ReadVariableOp^layer2/StatefulPartitionedCall-^layer2/kernel/Regularizer/Abs/ReadVariableOp^layer3/StatefulPartitionedCall-^layer3/kernel/Regularizer/Abs/ReadVariableOp^layer4/StatefulPartitionedCall-^layer4/kernel/Regularizer/Abs/ReadVariableOp^layer5/StatefulPartitionedCall-^layer5/kernel/Regularizer/Abs/ReadVariableOp^layer6/StatefulPartitionedCall-^layer6/kernel/Regularizer/Abs/ReadVariableOp^layer7/StatefulPartitionedCall-^layer7/kernel/Regularizer/Abs/ReadVariableOp^layer8/StatefulPartitionedCall-^layer8/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2\
,layer1/kernel/Regularizer/Abs/ReadVariableOp,layer1/kernel/Regularizer/Abs/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2\
,layer2/kernel/Regularizer/Abs/ReadVariableOp,layer2/kernel/Regularizer/Abs/ReadVariableOp2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2\
,layer3/kernel/Regularizer/Abs/ReadVariableOp,layer3/kernel/Regularizer/Abs/ReadVariableOp2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2\
,layer4/kernel/Regularizer/Abs/ReadVariableOp,layer4/kernel/Regularizer/Abs/ReadVariableOp2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2\
,layer5/kernel/Regularizer/Abs/ReadVariableOp,layer5/kernel/Regularizer/Abs/ReadVariableOp2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall2\
,layer6/kernel/Regularizer/Abs/ReadVariableOp,layer6/kernel/Regularizer/Abs/ReadVariableOp2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2\
,layer7/kernel/Regularizer/Abs/ReadVariableOp,layer7/kernel/Regularizer/Abs/ReadVariableOp2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2\
,layer8/kernel/Regularizer/Abs/ReadVariableOp,layer8/kernel/Regularizer/Abs/ReadVariableOp:P L
'
_output_shapes
:?????????/
!
_user_specified_name	input_9
?
?
B__inference_layer5_layer_call_and_return_conditional_losses_331754

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer5/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer5/kernel/Regularizer/Abs/ReadVariableOp?
layer5/kernel/Regularizer/AbsAbs4layer5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer5/kernel/Regularizer/Abs?
layer5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer5/kernel/Regularizer/Const?
layer5/kernel/Regularizer/SumSum!layer5/kernel/Regularizer/Abs:y:0(layer5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/Sum?
layer5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer5/kernel/Regularizer/mul/x?
layer5/kernel/Regularizer/mulMul(layer5/kernel/Regularizer/mul/x:output:0&layer5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer5/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer5/kernel/Regularizer/Abs/ReadVariableOp,layer5/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_331880G
5layer2_kernel_regularizer_abs_readvariableop_resource:/d
identity??,layer2/kernel/Regularizer/Abs/ReadVariableOp?
,layer2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5layer2_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:/d*
dtype02.
,layer2/kernel/Regularizer/Abs/ReadVariableOp?
layer2/kernel/Regularizer/AbsAbs4layer2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/d2
layer2/kernel/Regularizer/Abs?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer2/kernel/Regularizer/Const?
layer2/kernel/Regularizer/SumSum!layer2/kernel/Regularizer/Abs:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/Sum?
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer2/kernel/Regularizer/mul/x?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/mulk
IdentityIdentity!layer2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^layer2/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,layer2/kernel/Regularizer/Abs/ReadVariableOp,layer2/kernel/Regularizer/Abs/ReadVariableOp
?
?
B__inference_layer1_layer_call_and_return_conditional_losses_330505

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer1/kernel/Regularizer/Abs/ReadVariableOp?
layer1/kernel/Regularizer/AbsAbs4layer1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer1/kernel/Regularizer/Abs?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer1/kernel/Regularizer/Const?
layer1/kernel/Regularizer/SumSum!layer1/kernel/Regularizer/Abs:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/Sum?
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer1/kernel/Regularizer/mul/x?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer1/kernel/Regularizer/Abs/ReadVariableOp,layer1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
ߔ
?
H__inference_sequential_8_layer_call_and_return_conditional_losses_331422

inputs7
%layer1_matmul_readvariableop_resource://4
&layer1_biasadd_readvariableop_resource:/7
%layer2_matmul_readvariableop_resource:/d4
&layer2_biasadd_readvariableop_resource:d7
%layer3_matmul_readvariableop_resource:dd4
&layer3_biasadd_readvariableop_resource:d7
%layer4_matmul_readvariableop_resource:d/4
&layer4_biasadd_readvariableop_resource:/7
%layer5_matmul_readvariableop_resource://4
&layer5_biasadd_readvariableop_resource:/7
%layer6_matmul_readvariableop_resource://4
&layer6_biasadd_readvariableop_resource:/7
%layer7_matmul_readvariableop_resource://4
&layer7_biasadd_readvariableop_resource:/7
%layer8_matmul_readvariableop_resource:/4
&layer8_biasadd_readvariableop_resource:
identity??layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?,layer1/kernel/Regularizer/Abs/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/MatMul/ReadVariableOp?,layer2/kernel/Regularizer/Abs/ReadVariableOp?layer3/BiasAdd/ReadVariableOp?layer3/MatMul/ReadVariableOp?,layer3/kernel/Regularizer/Abs/ReadVariableOp?layer4/BiasAdd/ReadVariableOp?layer4/MatMul/ReadVariableOp?,layer4/kernel/Regularizer/Abs/ReadVariableOp?layer5/BiasAdd/ReadVariableOp?layer5/MatMul/ReadVariableOp?,layer5/kernel/Regularizer/Abs/ReadVariableOp?layer6/BiasAdd/ReadVariableOp?layer6/MatMul/ReadVariableOp?,layer6/kernel/Regularizer/Abs/ReadVariableOp?layer7/BiasAdd/ReadVariableOp?layer7/MatMul/ReadVariableOp?,layer7/kernel/Regularizer/Abs/ReadVariableOp?layer8/BiasAdd/ReadVariableOp?layer8/MatMul/ReadVariableOp?,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

://*
dtype02
layer1/MatMul/ReadVariableOp?
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer1/MatMul?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer1/BiasAddm
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer1/Relu?
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:/d*
dtype02
layer2/MatMul/ReadVariableOp?
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer2/MatMul?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
layer2/Relu?
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
layer3/MatMul/ReadVariableOp?
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer3/MatMul?
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer3/BiasAdd/ReadVariableOp?
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
layer3/Relu?
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes

:d/*
dtype02
layer4/MatMul/ReadVariableOp?
layer4/MatMulMatMullayer3/Relu:activations:0$layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer4/MatMul?
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer4/BiasAdd/ReadVariableOp?
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer4/BiasAddm
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer4/Relu?
layer5/MatMul/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

://*
dtype02
layer5/MatMul/ReadVariableOp?
layer5/MatMulMatMullayer4/Relu:activations:0$layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer5/MatMul?
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer5/BiasAdd/ReadVariableOp?
layer5/BiasAddBiasAddlayer5/MatMul:product:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer5/BiasAddm
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer5/Relu?
layer6/MatMul/ReadVariableOpReadVariableOp%layer6_matmul_readvariableop_resource*
_output_shapes

://*
dtype02
layer6/MatMul/ReadVariableOp?
layer6/MatMulMatMullayer5/Relu:activations:0$layer6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer6/MatMul?
layer6/BiasAdd/ReadVariableOpReadVariableOp&layer6_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer6/BiasAdd/ReadVariableOp?
layer6/BiasAddBiasAddlayer6/MatMul:product:0%layer6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer6/BiasAddm
layer6/ReluRelulayer6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer6/Relu?
layer7/MatMul/ReadVariableOpReadVariableOp%layer7_matmul_readvariableop_resource*
_output_shapes

://*
dtype02
layer7/MatMul/ReadVariableOp?
layer7/MatMulMatMullayer6/Relu:activations:0$layer7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer7/MatMul?
layer7/BiasAdd/ReadVariableOpReadVariableOp&layer7_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
layer7/BiasAdd/ReadVariableOp?
layer7/BiasAddBiasAddlayer7/MatMul:product:0%layer7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
layer7/BiasAddm
layer7/ReluRelulayer7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
layer7/Relu?
layer8/MatMul/ReadVariableOpReadVariableOp%layer8_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02
layer8/MatMul/ReadVariableOp?
layer8/MatMulMatMullayer7/Relu:activations:0$layer8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer8/MatMul?
layer8/BiasAdd/ReadVariableOpReadVariableOp&layer8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer8/BiasAdd/ReadVariableOp?
layer8/BiasAddBiasAddlayer8/MatMul:product:0%layer8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer8/BiasAdd?
,layer1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer1/kernel/Regularizer/Abs/ReadVariableOp?
layer1/kernel/Regularizer/AbsAbs4layer1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer1/kernel/Regularizer/Abs?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer1/kernel/Regularizer/Const?
layer1/kernel/Regularizer/SumSum!layer1/kernel/Regularizer/Abs:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/Sum?
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer1/kernel/Regularizer/mul/x?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/mul?
,layer2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:/d*
dtype02.
,layer2/kernel/Regularizer/Abs/ReadVariableOp?
layer2/kernel/Regularizer/AbsAbs4layer2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/d2
layer2/kernel/Regularizer/Abs?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer2/kernel/Regularizer/Const?
layer2/kernel/Regularizer/SumSum!layer2/kernel/Regularizer/Abs:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/Sum?
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer2/kernel/Regularizer/mul/x?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/mul?
,layer3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,layer3/kernel/Regularizer/Abs/ReadVariableOp?
layer3/kernel/Regularizer/AbsAbs4layer3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2
layer3/kernel/Regularizer/Abs?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer3/kernel/Regularizer/Const?
layer3/kernel/Regularizer/SumSum!layer3/kernel/Regularizer/Abs:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/Sum?
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer3/kernel/Regularizer/mul/x?
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/mul?
,layer4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes

:d/*
dtype02.
,layer4/kernel/Regularizer/Abs/ReadVariableOp?
layer4/kernel/Regularizer/AbsAbs4layer4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d/2
layer4/kernel/Regularizer/Abs?
layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer4/kernel/Regularizer/Const?
layer4/kernel/Regularizer/SumSum!layer4/kernel/Regularizer/Abs:y:0(layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/Sum?
layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer4/kernel/Regularizer/mul/x?
layer4/kernel/Regularizer/mulMul(layer4/kernel/Regularizer/mul/x:output:0&layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/mul?
,layer5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer5/kernel/Regularizer/Abs/ReadVariableOp?
layer5/kernel/Regularizer/AbsAbs4layer5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer5/kernel/Regularizer/Abs?
layer5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer5/kernel/Regularizer/Const?
layer5/kernel/Regularizer/SumSum!layer5/kernel/Regularizer/Abs:y:0(layer5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/Sum?
layer5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer5/kernel/Regularizer/mul/x?
layer5/kernel/Regularizer/mulMul(layer5/kernel/Regularizer/mul/x:output:0&layer5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/mul?
,layer6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer6_matmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer6/kernel/Regularizer/Abs/ReadVariableOp?
layer6/kernel/Regularizer/AbsAbs4layer6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer6/kernel/Regularizer/Abs?
layer6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer6/kernel/Regularizer/Const?
layer6/kernel/Regularizer/SumSum!layer6/kernel/Regularizer/Abs:y:0(layer6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/Sum?
layer6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer6/kernel/Regularizer/mul/x?
layer6/kernel/Regularizer/mulMul(layer6/kernel/Regularizer/mul/x:output:0&layer6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/mul?
,layer7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer7_matmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer7/kernel/Regularizer/Abs/ReadVariableOp?
layer7/kernel/Regularizer/AbsAbs4layer7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer7/kernel/Regularizer/Abs?
layer7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer7/kernel/Regularizer/Const?
layer7/kernel/Regularizer/SumSum!layer7/kernel/Regularizer/Abs:y:0(layer7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/Sum?
layer7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer7/kernel/Regularizer/mul/x?
layer7/kernel/Regularizer/mulMul(layer7/kernel/Regularizer/mul/x:output:0&layer7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/mul?
,layer8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%layer8_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02.
,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer8/kernel/Regularizer/AbsAbs4layer8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/2
layer8/kernel/Regularizer/Abs?
layer8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer8/kernel/Regularizer/Const?
layer8/kernel/Regularizer/SumSum!layer8/kernel/Regularizer/Abs:y:0(layer8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/Sum?
layer8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer8/kernel/Regularizer/mul/x?
layer8/kernel/Regularizer/mulMul(layer8/kernel/Regularizer/mul/x:output:0&layer8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/mulr
IdentityIdentitylayer8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp-^layer1/kernel/Regularizer/Abs/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp-^layer2/kernel/Regularizer/Abs/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp-^layer3/kernel/Regularizer/Abs/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp-^layer4/kernel/Regularizer/Abs/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/MatMul/ReadVariableOp-^layer5/kernel/Regularizer/Abs/ReadVariableOp^layer6/BiasAdd/ReadVariableOp^layer6/MatMul/ReadVariableOp-^layer6/kernel/Regularizer/Abs/ReadVariableOp^layer7/BiasAdd/ReadVariableOp^layer7/MatMul/ReadVariableOp-^layer7/kernel/Regularizer/Abs/ReadVariableOp^layer8/BiasAdd/ReadVariableOp^layer8/MatMul/ReadVariableOp-^layer8/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2\
,layer1/kernel/Regularizer/Abs/ReadVariableOp,layer1/kernel/Regularizer/Abs/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2\
,layer2/kernel/Regularizer/Abs/ReadVariableOp,layer2/kernel/Regularizer/Abs/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2\
,layer3/kernel/Regularizer/Abs/ReadVariableOp,layer3/kernel/Regularizer/Abs/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2\
,layer4/kernel/Regularizer/Abs/ReadVariableOp,layer4/kernel/Regularizer/Abs/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/MatMul/ReadVariableOplayer5/MatMul/ReadVariableOp2\
,layer5/kernel/Regularizer/Abs/ReadVariableOp,layer5/kernel/Regularizer/Abs/ReadVariableOp2>
layer6/BiasAdd/ReadVariableOplayer6/BiasAdd/ReadVariableOp2<
layer6/MatMul/ReadVariableOplayer6/MatMul/ReadVariableOp2\
,layer6/kernel/Regularizer/Abs/ReadVariableOp,layer6/kernel/Regularizer/Abs/ReadVariableOp2>
layer7/BiasAdd/ReadVariableOplayer7/BiasAdd/ReadVariableOp2<
layer7/MatMul/ReadVariableOplayer7/MatMul/ReadVariableOp2\
,layer7/kernel/Regularizer/Abs/ReadVariableOp,layer7/kernel/Regularizer/Abs/ReadVariableOp2>
layer8/BiasAdd/ReadVariableOplayer8/BiasAdd/ReadVariableOp2<
layer8/MatMul/ReadVariableOplayer8/MatMul/ReadVariableOp2\
,layer8/kernel/Regularizer/Abs/ReadVariableOp,layer8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_331315
input_9
unknown://
	unknown_0:/
	unknown_1:/d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d/
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_3304812
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????/
!
_user_specified_name	input_9
?^
?
!__inference__wrapped_model_330481
input_9D
2sequential_8_layer1_matmul_readvariableop_resource://A
3sequential_8_layer1_biasadd_readvariableop_resource:/D
2sequential_8_layer2_matmul_readvariableop_resource:/dA
3sequential_8_layer2_biasadd_readvariableop_resource:dD
2sequential_8_layer3_matmul_readvariableop_resource:ddA
3sequential_8_layer3_biasadd_readvariableop_resource:dD
2sequential_8_layer4_matmul_readvariableop_resource:d/A
3sequential_8_layer4_biasadd_readvariableop_resource:/D
2sequential_8_layer5_matmul_readvariableop_resource://A
3sequential_8_layer5_biasadd_readvariableop_resource:/D
2sequential_8_layer6_matmul_readvariableop_resource://A
3sequential_8_layer6_biasadd_readvariableop_resource:/D
2sequential_8_layer7_matmul_readvariableop_resource://A
3sequential_8_layer7_biasadd_readvariableop_resource:/D
2sequential_8_layer8_matmul_readvariableop_resource:/A
3sequential_8_layer8_biasadd_readvariableop_resource:
identity??*sequential_8/layer1/BiasAdd/ReadVariableOp?)sequential_8/layer1/MatMul/ReadVariableOp?*sequential_8/layer2/BiasAdd/ReadVariableOp?)sequential_8/layer2/MatMul/ReadVariableOp?*sequential_8/layer3/BiasAdd/ReadVariableOp?)sequential_8/layer3/MatMul/ReadVariableOp?*sequential_8/layer4/BiasAdd/ReadVariableOp?)sequential_8/layer4/MatMul/ReadVariableOp?*sequential_8/layer5/BiasAdd/ReadVariableOp?)sequential_8/layer5/MatMul/ReadVariableOp?*sequential_8/layer6/BiasAdd/ReadVariableOp?)sequential_8/layer6/MatMul/ReadVariableOp?*sequential_8/layer7/BiasAdd/ReadVariableOp?)sequential_8/layer7/MatMul/ReadVariableOp?*sequential_8/layer8/BiasAdd/ReadVariableOp?)sequential_8/layer8/MatMul/ReadVariableOp?
)sequential_8/layer1/MatMul/ReadVariableOpReadVariableOp2sequential_8_layer1_matmul_readvariableop_resource*
_output_shapes

://*
dtype02+
)sequential_8/layer1/MatMul/ReadVariableOp?
sequential_8/layer1/MatMulMatMulinput_91sequential_8/layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer1/MatMul?
*sequential_8/layer1/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_layer1_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02,
*sequential_8/layer1/BiasAdd/ReadVariableOp?
sequential_8/layer1/BiasAddBiasAdd$sequential_8/layer1/MatMul:product:02sequential_8/layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer1/BiasAdd?
sequential_8/layer1/ReluRelu$sequential_8/layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer1/Relu?
)sequential_8/layer2/MatMul/ReadVariableOpReadVariableOp2sequential_8_layer2_matmul_readvariableop_resource*
_output_shapes

:/d*
dtype02+
)sequential_8/layer2/MatMul/ReadVariableOp?
sequential_8/layer2/MatMulMatMul&sequential_8/layer1/Relu:activations:01sequential_8/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_8/layer2/MatMul?
*sequential_8/layer2/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02,
*sequential_8/layer2/BiasAdd/ReadVariableOp?
sequential_8/layer2/BiasAddBiasAdd$sequential_8/layer2/MatMul:product:02sequential_8/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_8/layer2/BiasAdd?
sequential_8/layer2/ReluRelu$sequential_8/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_8/layer2/Relu?
)sequential_8/layer3/MatMul/ReadVariableOpReadVariableOp2sequential_8_layer3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02+
)sequential_8/layer3/MatMul/ReadVariableOp?
sequential_8/layer3/MatMulMatMul&sequential_8/layer2/Relu:activations:01sequential_8/layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_8/layer3/MatMul?
*sequential_8/layer3/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_layer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02,
*sequential_8/layer3/BiasAdd/ReadVariableOp?
sequential_8/layer3/BiasAddBiasAdd$sequential_8/layer3/MatMul:product:02sequential_8/layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_8/layer3/BiasAdd?
sequential_8/layer3/ReluRelu$sequential_8/layer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_8/layer3/Relu?
)sequential_8/layer4/MatMul/ReadVariableOpReadVariableOp2sequential_8_layer4_matmul_readvariableop_resource*
_output_shapes

:d/*
dtype02+
)sequential_8/layer4/MatMul/ReadVariableOp?
sequential_8/layer4/MatMulMatMul&sequential_8/layer3/Relu:activations:01sequential_8/layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer4/MatMul?
*sequential_8/layer4/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_layer4_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02,
*sequential_8/layer4/BiasAdd/ReadVariableOp?
sequential_8/layer4/BiasAddBiasAdd$sequential_8/layer4/MatMul:product:02sequential_8/layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer4/BiasAdd?
sequential_8/layer4/ReluRelu$sequential_8/layer4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer4/Relu?
)sequential_8/layer5/MatMul/ReadVariableOpReadVariableOp2sequential_8_layer5_matmul_readvariableop_resource*
_output_shapes

://*
dtype02+
)sequential_8/layer5/MatMul/ReadVariableOp?
sequential_8/layer5/MatMulMatMul&sequential_8/layer4/Relu:activations:01sequential_8/layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer5/MatMul?
*sequential_8/layer5/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_layer5_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02,
*sequential_8/layer5/BiasAdd/ReadVariableOp?
sequential_8/layer5/BiasAddBiasAdd$sequential_8/layer5/MatMul:product:02sequential_8/layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer5/BiasAdd?
sequential_8/layer5/ReluRelu$sequential_8/layer5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer5/Relu?
)sequential_8/layer6/MatMul/ReadVariableOpReadVariableOp2sequential_8_layer6_matmul_readvariableop_resource*
_output_shapes

://*
dtype02+
)sequential_8/layer6/MatMul/ReadVariableOp?
sequential_8/layer6/MatMulMatMul&sequential_8/layer5/Relu:activations:01sequential_8/layer6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer6/MatMul?
*sequential_8/layer6/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_layer6_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02,
*sequential_8/layer6/BiasAdd/ReadVariableOp?
sequential_8/layer6/BiasAddBiasAdd$sequential_8/layer6/MatMul:product:02sequential_8/layer6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer6/BiasAdd?
sequential_8/layer6/ReluRelu$sequential_8/layer6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer6/Relu?
)sequential_8/layer7/MatMul/ReadVariableOpReadVariableOp2sequential_8_layer7_matmul_readvariableop_resource*
_output_shapes

://*
dtype02+
)sequential_8/layer7/MatMul/ReadVariableOp?
sequential_8/layer7/MatMulMatMul&sequential_8/layer6/Relu:activations:01sequential_8/layer7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer7/MatMul?
*sequential_8/layer7/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_layer7_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02,
*sequential_8/layer7/BiasAdd/ReadVariableOp?
sequential_8/layer7/BiasAddBiasAdd$sequential_8/layer7/MatMul:product:02sequential_8/layer7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer7/BiasAdd?
sequential_8/layer7/ReluRelu$sequential_8/layer7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
sequential_8/layer7/Relu?
)sequential_8/layer8/MatMul/ReadVariableOpReadVariableOp2sequential_8_layer8_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02+
)sequential_8/layer8/MatMul/ReadVariableOp?
sequential_8/layer8/MatMulMatMul&sequential_8/layer7/Relu:activations:01sequential_8/layer8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/layer8/MatMul?
*sequential_8/layer8/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_layer8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_8/layer8/BiasAdd/ReadVariableOp?
sequential_8/layer8/BiasAddBiasAdd$sequential_8/layer8/MatMul:product:02sequential_8/layer8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/layer8/BiasAdd
IdentityIdentity$sequential_8/layer8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp+^sequential_8/layer1/BiasAdd/ReadVariableOp*^sequential_8/layer1/MatMul/ReadVariableOp+^sequential_8/layer2/BiasAdd/ReadVariableOp*^sequential_8/layer2/MatMul/ReadVariableOp+^sequential_8/layer3/BiasAdd/ReadVariableOp*^sequential_8/layer3/MatMul/ReadVariableOp+^sequential_8/layer4/BiasAdd/ReadVariableOp*^sequential_8/layer4/MatMul/ReadVariableOp+^sequential_8/layer5/BiasAdd/ReadVariableOp*^sequential_8/layer5/MatMul/ReadVariableOp+^sequential_8/layer6/BiasAdd/ReadVariableOp*^sequential_8/layer6/MatMul/ReadVariableOp+^sequential_8/layer7/BiasAdd/ReadVariableOp*^sequential_8/layer7/MatMul/ReadVariableOp+^sequential_8/layer8/BiasAdd/ReadVariableOp*^sequential_8/layer8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 2X
*sequential_8/layer1/BiasAdd/ReadVariableOp*sequential_8/layer1/BiasAdd/ReadVariableOp2V
)sequential_8/layer1/MatMul/ReadVariableOp)sequential_8/layer1/MatMul/ReadVariableOp2X
*sequential_8/layer2/BiasAdd/ReadVariableOp*sequential_8/layer2/BiasAdd/ReadVariableOp2V
)sequential_8/layer2/MatMul/ReadVariableOp)sequential_8/layer2/MatMul/ReadVariableOp2X
*sequential_8/layer3/BiasAdd/ReadVariableOp*sequential_8/layer3/BiasAdd/ReadVariableOp2V
)sequential_8/layer3/MatMul/ReadVariableOp)sequential_8/layer3/MatMul/ReadVariableOp2X
*sequential_8/layer4/BiasAdd/ReadVariableOp*sequential_8/layer4/BiasAdd/ReadVariableOp2V
)sequential_8/layer4/MatMul/ReadVariableOp)sequential_8/layer4/MatMul/ReadVariableOp2X
*sequential_8/layer5/BiasAdd/ReadVariableOp*sequential_8/layer5/BiasAdd/ReadVariableOp2V
)sequential_8/layer5/MatMul/ReadVariableOp)sequential_8/layer5/MatMul/ReadVariableOp2X
*sequential_8/layer6/BiasAdd/ReadVariableOp*sequential_8/layer6/BiasAdd/ReadVariableOp2V
)sequential_8/layer6/MatMul/ReadVariableOp)sequential_8/layer6/MatMul/ReadVariableOp2X
*sequential_8/layer7/BiasAdd/ReadVariableOp*sequential_8/layer7/BiasAdd/ReadVariableOp2V
)sequential_8/layer7/MatMul/ReadVariableOp)sequential_8/layer7/MatMul/ReadVariableOp2X
*sequential_8/layer8/BiasAdd/ReadVariableOp*sequential_8/layer8/BiasAdd/ReadVariableOp2V
)sequential_8/layer8/MatMul/ReadVariableOp)sequential_8/layer8/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????/
!
_user_specified_name	input_9
?
?
B__inference_layer5_layer_call_and_return_conditional_losses_330597

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer5/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer5/kernel/Regularizer/Abs/ReadVariableOp?
layer5/kernel/Regularizer/AbsAbs4layer5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer5/kernel/Regularizer/Abs?
layer5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer5/kernel/Regularizer/Const?
layer5/kernel/Regularizer/SumSum!layer5/kernel/Regularizer/Abs:y:0(layer5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/Sum?
layer5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer5/kernel/Regularizer/mul/x?
layer5/kernel/Regularizer/mulMul(layer5/kernel/Regularizer/mul/x:output:0&layer5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer5/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer5/kernel/Regularizer/Abs/ReadVariableOp,layer5/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
B__inference_layer4_layer_call_and_return_conditional_losses_330574

inputs0
matmul_readvariableop_resource:d/-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer4/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d/*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d/*
dtype02.
,layer4/kernel/Regularizer/Abs/ReadVariableOp?
layer4/kernel/Regularizer/AbsAbs4layer4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d/2
layer4/kernel/Regularizer/Abs?
layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer4/kernel/Regularizer/Const?
layer4/kernel/Regularizer/SumSum!layer4/kernel/Regularizer/Abs:y:0(layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/Sum?
layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer4/kernel/Regularizer/mul/x?
layer4/kernel/Regularizer/mulMul(layer4/kernel/Regularizer/mul/x:output:0&layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer4/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer4/kernel/Regularizer/Abs/ReadVariableOp,layer4/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_331891G
5layer3_kernel_regularizer_abs_readvariableop_resource:dd
identity??,layer3/kernel/Regularizer/Abs/ReadVariableOp?
,layer3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5layer3_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,layer3/kernel/Regularizer/Abs/ReadVariableOp?
layer3/kernel/Regularizer/AbsAbs4layer3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2
layer3/kernel/Regularizer/Abs?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer3/kernel/Regularizer/Const?
layer3/kernel/Regularizer/SumSum!layer3/kernel/Regularizer/Abs:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/Sum?
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer3/kernel/Regularizer/mul/x?
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/mulk
IdentityIdentity!layer3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^layer3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,layer3/kernel/Regularizer/Abs/ReadVariableOp,layer3/kernel/Regularizer/Abs/ReadVariableOp
?
?
'__inference_layer2_layer_call_fn_331667

inputs
unknown:/d
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_3305282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?u
?	
H__inference_sequential_8_layer_call_and_return_conditional_losses_330720

inputs
layer1_330506://
layer1_330508:/
layer2_330529:/d
layer2_330531:d
layer3_330552:dd
layer3_330554:d
layer4_330575:d/
layer4_330577:/
layer5_330598://
layer5_330600:/
layer6_330621://
layer6_330623:/
layer7_330644://
layer7_330646:/
layer8_330666:/
layer8_330668:
identity??layer1/StatefulPartitionedCall?,layer1/kernel/Regularizer/Abs/ReadVariableOp?layer2/StatefulPartitionedCall?,layer2/kernel/Regularizer/Abs/ReadVariableOp?layer3/StatefulPartitionedCall?,layer3/kernel/Regularizer/Abs/ReadVariableOp?layer4/StatefulPartitionedCall?,layer4/kernel/Regularizer/Abs/ReadVariableOp?layer5/StatefulPartitionedCall?,layer5/kernel/Regularizer/Abs/ReadVariableOp?layer6/StatefulPartitionedCall?,layer6/kernel/Regularizer/Abs/ReadVariableOp?layer7/StatefulPartitionedCall?,layer7/kernel/Regularizer/Abs/ReadVariableOp?layer8/StatefulPartitionedCall?,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_330506layer1_330508*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_3305052 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_330529layer2_330531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_3305282 
layer2/StatefulPartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_330552layer3_330554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_3305512 
layer3/StatefulPartitionedCall?
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_330575layer4_330577*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_3305742 
layer4/StatefulPartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_330598layer5_330600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_3305972 
layer5/StatefulPartitionedCall?
layer6/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0layer6_330621layer6_330623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_3306202 
layer6/StatefulPartitionedCall?
layer7/StatefulPartitionedCallStatefulPartitionedCall'layer6/StatefulPartitionedCall:output:0layer7_330644layer7_330646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_3306432 
layer7/StatefulPartitionedCall?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_330666layer8_330668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_3306652 
layer8/StatefulPartitionedCall?
,layer1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer1_330506*
_output_shapes

://*
dtype02.
,layer1/kernel/Regularizer/Abs/ReadVariableOp?
layer1/kernel/Regularizer/AbsAbs4layer1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer1/kernel/Regularizer/Abs?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer1/kernel/Regularizer/Const?
layer1/kernel/Regularizer/SumSum!layer1/kernel/Regularizer/Abs:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/Sum?
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer1/kernel/Regularizer/mul/x?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/mul?
,layer2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer2_330529*
_output_shapes

:/d*
dtype02.
,layer2/kernel/Regularizer/Abs/ReadVariableOp?
layer2/kernel/Regularizer/AbsAbs4layer2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/d2
layer2/kernel/Regularizer/Abs?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer2/kernel/Regularizer/Const?
layer2/kernel/Regularizer/SumSum!layer2/kernel/Regularizer/Abs:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/Sum?
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer2/kernel/Regularizer/mul/x?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/mul?
,layer3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer3_330552*
_output_shapes

:dd*
dtype02.
,layer3/kernel/Regularizer/Abs/ReadVariableOp?
layer3/kernel/Regularizer/AbsAbs4layer3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2
layer3/kernel/Regularizer/Abs?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer3/kernel/Regularizer/Const?
layer3/kernel/Regularizer/SumSum!layer3/kernel/Regularizer/Abs:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/Sum?
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer3/kernel/Regularizer/mul/x?
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/mul?
,layer4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer4_330575*
_output_shapes

:d/*
dtype02.
,layer4/kernel/Regularizer/Abs/ReadVariableOp?
layer4/kernel/Regularizer/AbsAbs4layer4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d/2
layer4/kernel/Regularizer/Abs?
layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer4/kernel/Regularizer/Const?
layer4/kernel/Regularizer/SumSum!layer4/kernel/Regularizer/Abs:y:0(layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/Sum?
layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer4/kernel/Regularizer/mul/x?
layer4/kernel/Regularizer/mulMul(layer4/kernel/Regularizer/mul/x:output:0&layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/mul?
,layer5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer5_330598*
_output_shapes

://*
dtype02.
,layer5/kernel/Regularizer/Abs/ReadVariableOp?
layer5/kernel/Regularizer/AbsAbs4layer5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer5/kernel/Regularizer/Abs?
layer5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer5/kernel/Regularizer/Const?
layer5/kernel/Regularizer/SumSum!layer5/kernel/Regularizer/Abs:y:0(layer5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/Sum?
layer5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer5/kernel/Regularizer/mul/x?
layer5/kernel/Regularizer/mulMul(layer5/kernel/Regularizer/mul/x:output:0&layer5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/mul?
,layer6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer6_330621*
_output_shapes

://*
dtype02.
,layer6/kernel/Regularizer/Abs/ReadVariableOp?
layer6/kernel/Regularizer/AbsAbs4layer6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer6/kernel/Regularizer/Abs?
layer6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer6/kernel/Regularizer/Const?
layer6/kernel/Regularizer/SumSum!layer6/kernel/Regularizer/Abs:y:0(layer6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/Sum?
layer6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer6/kernel/Regularizer/mul/x?
layer6/kernel/Regularizer/mulMul(layer6/kernel/Regularizer/mul/x:output:0&layer6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/mul?
,layer7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer7_330644*
_output_shapes

://*
dtype02.
,layer7/kernel/Regularizer/Abs/ReadVariableOp?
layer7/kernel/Regularizer/AbsAbs4layer7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer7/kernel/Regularizer/Abs?
layer7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer7/kernel/Regularizer/Const?
layer7/kernel/Regularizer/SumSum!layer7/kernel/Regularizer/Abs:y:0(layer7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/Sum?
layer7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer7/kernel/Regularizer/mul/x?
layer7/kernel/Regularizer/mulMul(layer7/kernel/Regularizer/mul/x:output:0&layer7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/mul?
,layer8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer8_330666*
_output_shapes

:/*
dtype02.
,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer8/kernel/Regularizer/AbsAbs4layer8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/2
layer8/kernel/Regularizer/Abs?
layer8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer8/kernel/Regularizer/Const?
layer8/kernel/Regularizer/SumSum!layer8/kernel/Regularizer/Abs:y:0(layer8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/Sum?
layer8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer8/kernel/Regularizer/mul/x?
layer8/kernel/Regularizer/mulMul(layer8/kernel/Regularizer/mul/x:output:0&layer8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/mul?
IdentityIdentity'layer8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^layer1/StatefulPartitionedCall-^layer1/kernel/Regularizer/Abs/ReadVariableOp^layer2/StatefulPartitionedCall-^layer2/kernel/Regularizer/Abs/ReadVariableOp^layer3/StatefulPartitionedCall-^layer3/kernel/Regularizer/Abs/ReadVariableOp^layer4/StatefulPartitionedCall-^layer4/kernel/Regularizer/Abs/ReadVariableOp^layer5/StatefulPartitionedCall-^layer5/kernel/Regularizer/Abs/ReadVariableOp^layer6/StatefulPartitionedCall-^layer6/kernel/Regularizer/Abs/ReadVariableOp^layer7/StatefulPartitionedCall-^layer7/kernel/Regularizer/Abs/ReadVariableOp^layer8/StatefulPartitionedCall-^layer8/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2\
,layer1/kernel/Regularizer/Abs/ReadVariableOp,layer1/kernel/Regularizer/Abs/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2\
,layer2/kernel/Regularizer/Abs/ReadVariableOp,layer2/kernel/Regularizer/Abs/ReadVariableOp2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2\
,layer3/kernel/Regularizer/Abs/ReadVariableOp,layer3/kernel/Regularizer/Abs/ReadVariableOp2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2\
,layer4/kernel/Regularizer/Abs/ReadVariableOp,layer4/kernel/Regularizer/Abs/ReadVariableOp2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2\
,layer5/kernel/Regularizer/Abs/ReadVariableOp,layer5/kernel/Regularizer/Abs/ReadVariableOp2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall2\
,layer6/kernel/Regularizer/Abs/ReadVariableOp,layer6/kernel/Regularizer/Abs/ReadVariableOp2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2\
,layer7/kernel/Regularizer/Abs/ReadVariableOp,layer7/kernel/Regularizer/Abs/ReadVariableOp2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2\
,layer8/kernel/Regularizer/Abs/ReadVariableOp,layer8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
'__inference_layer1_layer_call_fn_331635

inputs
unknown://
	unknown_0:/
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_3305052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
'__inference_layer3_layer_call_fn_331699

inputs
unknown:dd
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_3305512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_331924G
5layer6_kernel_regularizer_abs_readvariableop_resource://
identity??,layer6/kernel/Regularizer/Abs/ReadVariableOp?
,layer6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5layer6_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer6/kernel/Regularizer/Abs/ReadVariableOp?
layer6/kernel/Regularizer/AbsAbs4layer6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer6/kernel/Regularizer/Abs?
layer6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer6/kernel/Regularizer/Const?
layer6/kernel/Regularizer/SumSum!layer6/kernel/Regularizer/Abs:y:0(layer6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/Sum?
layer6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer6/kernel/Regularizer/mul/x?
layer6/kernel/Regularizer/mulMul(layer6/kernel/Regularizer/mul/x:output:0&layer6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/mulk
IdentityIdentity!layer6/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^layer6/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,layer6/kernel/Regularizer/Abs/ReadVariableOp,layer6/kernel/Regularizer/Abs/ReadVariableOp
?
?
__inference_loss_fn_7_331946G
5layer8_kernel_regularizer_abs_readvariableop_resource:/
identity??,layer8/kernel/Regularizer/Abs/ReadVariableOp?
,layer8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5layer8_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:/*
dtype02.
,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer8/kernel/Regularizer/AbsAbs4layer8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/2
layer8/kernel/Regularizer/Abs?
layer8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer8/kernel/Regularizer/Const?
layer8/kernel/Regularizer/SumSum!layer8/kernel/Regularizer/Abs:y:0(layer8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/Sum?
layer8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer8/kernel/Regularizer/mul/x?
layer8/kernel/Regularizer/mulMul(layer8/kernel/Regularizer/mul/x:output:0&layer8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/mulk
IdentityIdentity!layer8/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^layer8/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,layer8/kernel/Regularizer/Abs/ReadVariableOp,layer8/kernel/Regularizer/Abs/ReadVariableOp
?
?
'__inference_layer5_layer_call_fn_331763

inputs
unknown://
	unknown_0:/
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_3305972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
-__inference_sequential_8_layer_call_fn_331566

inputs
unknown://
	unknown_0:/
	unknown_1:/d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d/
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_3307202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
'__inference_layer6_layer_call_fn_331795

inputs
unknown://
	unknown_0:/
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_3306202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
-__inference_sequential_8_layer_call_fn_330755
input_9
unknown://
	unknown_0:/
	unknown_1:/d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d/
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_3307202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????/
!
_user_specified_name	input_9
?
?
B__inference_layer8_layer_call_and_return_conditional_losses_331849

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer8/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
,layer8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype02.
,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer8/kernel/Regularizer/AbsAbs4layer8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/2
layer8/kernel/Regularizer/Abs?
layer8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer8/kernel/Regularizer/Const?
layer8/kernel/Regularizer/SumSum!layer8/kernel/Regularizer/Abs:y:0(layer8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/Sum?
layer8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer8/kernel/Regularizer/mul/x?
layer8/kernel/Regularizer/mulMul(layer8/kernel/Regularizer/mul/x:output:0&layer8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer8/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer8/kernel/Regularizer/Abs/ReadVariableOp,layer8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_331902G
5layer4_kernel_regularizer_abs_readvariableop_resource:d/
identity??,layer4/kernel/Regularizer/Abs/ReadVariableOp?
,layer4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5layer4_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d/*
dtype02.
,layer4/kernel/Regularizer/Abs/ReadVariableOp?
layer4/kernel/Regularizer/AbsAbs4layer4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d/2
layer4/kernel/Regularizer/Abs?
layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer4/kernel/Regularizer/Const?
layer4/kernel/Regularizer/SumSum!layer4/kernel/Regularizer/Abs:y:0(layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/Sum?
layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer4/kernel/Regularizer/mul/x?
layer4/kernel/Regularizer/mulMul(layer4/kernel/Regularizer/mul/x:output:0&layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/mulk
IdentityIdentity!layer4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^layer4/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,layer4/kernel/Regularizer/Abs/ReadVariableOp,layer4/kernel/Regularizer/Abs/ReadVariableOp
?
?
B__inference_layer6_layer_call_and_return_conditional_losses_330620

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer6/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/2
Relu?
,layer6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer6/kernel/Regularizer/Abs/ReadVariableOp?
layer6/kernel/Regularizer/AbsAbs4layer6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer6/kernel/Regularizer/Abs?
layer6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer6/kernel/Regularizer/Const?
layer6/kernel/Regularizer/SumSum!layer6/kernel/Regularizer/Abs:y:0(layer6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/Sum?
layer6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer6/kernel/Regularizer/mul/x?
layer6/kernel/Regularizer/mulMul(layer6/kernel/Regularizer/mul/x:output:0&layer6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer6/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer6/kernel/Regularizer/Abs/ReadVariableOp,layer6/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
-__inference_sequential_8_layer_call_fn_331038
input_9
unknown://
	unknown_0:/
	unknown_1:/d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d/
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_3309662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????/
!
_user_specified_name	input_9
?u
?	
H__inference_sequential_8_layer_call_and_return_conditional_losses_330966

inputs
layer1_330877://
layer1_330879:/
layer2_330882:/d
layer2_330884:d
layer3_330887:dd
layer3_330889:d
layer4_330892:d/
layer4_330894:/
layer5_330897://
layer5_330899:/
layer6_330902://
layer6_330904:/
layer7_330907://
layer7_330909:/
layer8_330912:/
layer8_330914:
identity??layer1/StatefulPartitionedCall?,layer1/kernel/Regularizer/Abs/ReadVariableOp?layer2/StatefulPartitionedCall?,layer2/kernel/Regularizer/Abs/ReadVariableOp?layer3/StatefulPartitionedCall?,layer3/kernel/Regularizer/Abs/ReadVariableOp?layer4/StatefulPartitionedCall?,layer4/kernel/Regularizer/Abs/ReadVariableOp?layer5/StatefulPartitionedCall?,layer5/kernel/Regularizer/Abs/ReadVariableOp?layer6/StatefulPartitionedCall?,layer6/kernel/Regularizer/Abs/ReadVariableOp?layer7/StatefulPartitionedCall?,layer7/kernel/Regularizer/Abs/ReadVariableOp?layer8/StatefulPartitionedCall?,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_330877layer1_330879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_3305052 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_330882layer2_330884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_3305282 
layer2/StatefulPartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_330887layer3_330889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_3305512 
layer3/StatefulPartitionedCall?
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_330892layer4_330894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_3305742 
layer4/StatefulPartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_330897layer5_330899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_3305972 
layer5/StatefulPartitionedCall?
layer6/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0layer6_330902layer6_330904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_3306202 
layer6/StatefulPartitionedCall?
layer7/StatefulPartitionedCallStatefulPartitionedCall'layer6/StatefulPartitionedCall:output:0layer7_330907layer7_330909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_3306432 
layer7/StatefulPartitionedCall?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_330912layer8_330914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_3306652 
layer8/StatefulPartitionedCall?
,layer1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer1_330877*
_output_shapes

://*
dtype02.
,layer1/kernel/Regularizer/Abs/ReadVariableOp?
layer1/kernel/Regularizer/AbsAbs4layer1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer1/kernel/Regularizer/Abs?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer1/kernel/Regularizer/Const?
layer1/kernel/Regularizer/SumSum!layer1/kernel/Regularizer/Abs:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/Sum?
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer1/kernel/Regularizer/mul/x?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/mul?
,layer2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer2_330882*
_output_shapes

:/d*
dtype02.
,layer2/kernel/Regularizer/Abs/ReadVariableOp?
layer2/kernel/Regularizer/AbsAbs4layer2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/d2
layer2/kernel/Regularizer/Abs?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer2/kernel/Regularizer/Const?
layer2/kernel/Regularizer/SumSum!layer2/kernel/Regularizer/Abs:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/Sum?
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer2/kernel/Regularizer/mul/x?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer2/kernel/Regularizer/mul?
,layer3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer3_330887*
_output_shapes

:dd*
dtype02.
,layer3/kernel/Regularizer/Abs/ReadVariableOp?
layer3/kernel/Regularizer/AbsAbs4layer3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2
layer3/kernel/Regularizer/Abs?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer3/kernel/Regularizer/Const?
layer3/kernel/Regularizer/SumSum!layer3/kernel/Regularizer/Abs:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/Sum?
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer3/kernel/Regularizer/mul/x?
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/mul?
,layer4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer4_330892*
_output_shapes

:d/*
dtype02.
,layer4/kernel/Regularizer/Abs/ReadVariableOp?
layer4/kernel/Regularizer/AbsAbs4layer4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d/2
layer4/kernel/Regularizer/Abs?
layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer4/kernel/Regularizer/Const?
layer4/kernel/Regularizer/SumSum!layer4/kernel/Regularizer/Abs:y:0(layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/Sum?
layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer4/kernel/Regularizer/mul/x?
layer4/kernel/Regularizer/mulMul(layer4/kernel/Regularizer/mul/x:output:0&layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer4/kernel/Regularizer/mul?
,layer5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer5_330897*
_output_shapes

://*
dtype02.
,layer5/kernel/Regularizer/Abs/ReadVariableOp?
layer5/kernel/Regularizer/AbsAbs4layer5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer5/kernel/Regularizer/Abs?
layer5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer5/kernel/Regularizer/Const?
layer5/kernel/Regularizer/SumSum!layer5/kernel/Regularizer/Abs:y:0(layer5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/Sum?
layer5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer5/kernel/Regularizer/mul/x?
layer5/kernel/Regularizer/mulMul(layer5/kernel/Regularizer/mul/x:output:0&layer5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer5/kernel/Regularizer/mul?
,layer6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer6_330902*
_output_shapes

://*
dtype02.
,layer6/kernel/Regularizer/Abs/ReadVariableOp?
layer6/kernel/Regularizer/AbsAbs4layer6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer6/kernel/Regularizer/Abs?
layer6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer6/kernel/Regularizer/Const?
layer6/kernel/Regularizer/SumSum!layer6/kernel/Regularizer/Abs:y:0(layer6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/Sum?
layer6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer6/kernel/Regularizer/mul/x?
layer6/kernel/Regularizer/mulMul(layer6/kernel/Regularizer/mul/x:output:0&layer6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer6/kernel/Regularizer/mul?
,layer7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer7_330907*
_output_shapes

://*
dtype02.
,layer7/kernel/Regularizer/Abs/ReadVariableOp?
layer7/kernel/Regularizer/AbsAbs4layer7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer7/kernel/Regularizer/Abs?
layer7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer7/kernel/Regularizer/Const?
layer7/kernel/Regularizer/SumSum!layer7/kernel/Regularizer/Abs:y:0(layer7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/Sum?
layer7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer7/kernel/Regularizer/mul/x?
layer7/kernel/Regularizer/mulMul(layer7/kernel/Regularizer/mul/x:output:0&layer7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer7/kernel/Regularizer/mul?
,layer8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplayer8_330912*
_output_shapes

:/*
dtype02.
,layer8/kernel/Regularizer/Abs/ReadVariableOp?
layer8/kernel/Regularizer/AbsAbs4layer8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:/2
layer8/kernel/Regularizer/Abs?
layer8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer8/kernel/Regularizer/Const?
layer8/kernel/Regularizer/SumSum!layer8/kernel/Regularizer/Abs:y:0(layer8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/Sum?
layer8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer8/kernel/Regularizer/mul/x?
layer8/kernel/Regularizer/mulMul(layer8/kernel/Regularizer/mul/x:output:0&layer8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer8/kernel/Regularizer/mul?
IdentityIdentity'layer8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^layer1/StatefulPartitionedCall-^layer1/kernel/Regularizer/Abs/ReadVariableOp^layer2/StatefulPartitionedCall-^layer2/kernel/Regularizer/Abs/ReadVariableOp^layer3/StatefulPartitionedCall-^layer3/kernel/Regularizer/Abs/ReadVariableOp^layer4/StatefulPartitionedCall-^layer4/kernel/Regularizer/Abs/ReadVariableOp^layer5/StatefulPartitionedCall-^layer5/kernel/Regularizer/Abs/ReadVariableOp^layer6/StatefulPartitionedCall-^layer6/kernel/Regularizer/Abs/ReadVariableOp^layer7/StatefulPartitionedCall-^layer7/kernel/Regularizer/Abs/ReadVariableOp^layer8/StatefulPartitionedCall-^layer8/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????/: : : : : : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2\
,layer1/kernel/Regularizer/Abs/ReadVariableOp,layer1/kernel/Regularizer/Abs/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2\
,layer2/kernel/Regularizer/Abs/ReadVariableOp,layer2/kernel/Regularizer/Abs/ReadVariableOp2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2\
,layer3/kernel/Regularizer/Abs/ReadVariableOp,layer3/kernel/Regularizer/Abs/ReadVariableOp2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2\
,layer4/kernel/Regularizer/Abs/ReadVariableOp,layer4/kernel/Regularizer/Abs/ReadVariableOp2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2\
,layer5/kernel/Regularizer/Abs/ReadVariableOp,layer5/kernel/Regularizer/Abs/ReadVariableOp2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall2\
,layer6/kernel/Regularizer/Abs/ReadVariableOp,layer6/kernel/Regularizer/Abs/ReadVariableOp2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2\
,layer7/kernel/Regularizer/Abs/ReadVariableOp,layer7/kernel/Regularizer/Abs/ReadVariableOp2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2\
,layer8/kernel/Regularizer/Abs/ReadVariableOp,layer8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
B__inference_layer3_layer_call_and_return_conditional_losses_331690

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?,layer3/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
,layer3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,layer3/kernel/Regularizer/Abs/ReadVariableOp?
layer3/kernel/Regularizer/AbsAbs4layer3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:dd2
layer3/kernel/Regularizer/Abs?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer3/kernel/Regularizer/Const?
layer3/kernel/Regularizer/SumSum!layer3/kernel/Regularizer/Abs:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/Sum?
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer3/kernel/Regularizer/mul/x?
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer3/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^layer3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,layer3/kernel/Regularizer/Abs/ReadVariableOp,layer3/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_331869G
5layer1_kernel_regularizer_abs_readvariableop_resource://
identity??,layer1/kernel/Regularizer/Abs/ReadVariableOp?
,layer1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5layer1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

://*
dtype02.
,layer1/kernel/Regularizer/Abs/ReadVariableOp?
layer1/kernel/Regularizer/AbsAbs4layer1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

://2
layer1/kernel/Regularizer/Abs?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
layer1/kernel/Regularizer/Const?
layer1/kernel/Regularizer/SumSum!layer1/kernel/Regularizer/Abs:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/Sum?
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
layer1/kernel/Regularizer/mul/x?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
layer1/kernel/Regularizer/mulk
IdentityIdentity!layer1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^layer1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,layer1/kernel/Regularizer/Abs/ReadVariableOp,layer1/kernel/Regularizer/Abs/ReadVariableOp"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_90
serving_default_input_9:0?????????/:
layer80
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"
_tf_keras_sequential
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemvmwmxmymzm{!m|"m}'m~(m-m?.m?3m?4m?9m?:m?v?v?v?v?v?v?!v?"v?'v?(v?-v?.v?3v?4v?9v?:v?"
	optimizer
?
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15"
trackable_list_wrapper
`
?0
?1
?2
?3
?4
?5
?6
?7"
trackable_list_wrapper
?
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15"
trackable_list_wrapper
?

Dlayers
Elayer_regularization_losses
Fmetrics

trainable_variables
Gnon_trainable_variables
regularization_losses
	variables
Hlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
://2layer1/kernel
:/2layer1/bias
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Ilayers
Jlayer_regularization_losses
Kmetrics
trainable_variables
Lnon_trainable_variables
regularization_losses
	variables
Mlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:/d2layer2/kernel
:d2layer2/bias
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Nlayers
Olayer_regularization_losses
Pmetrics
trainable_variables
Qnon_trainable_variables
regularization_losses
	variables
Rlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:dd2layer3/kernel
:d2layer3/bias
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Slayers
Tlayer_regularization_losses
Umetrics
trainable_variables
Vnon_trainable_variables
regularization_losses
	variables
Wlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:d/2layer4/kernel
:/2layer4/bias
.
!0
"1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?

Xlayers
Ylayer_regularization_losses
Zmetrics
#trainable_variables
[non_trainable_variables
$regularization_losses
%	variables
\layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
://2layer5/kernel
:/2layer5/bias
.
'0
(1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?

]layers
^layer_regularization_losses
_metrics
)trainable_variables
`non_trainable_variables
*regularization_losses
+	variables
alayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
://2layer6/kernel
:/2layer6/bias
.
-0
.1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?

blayers
clayer_regularization_losses
dmetrics
/trainable_variables
enon_trainable_variables
0regularization_losses
1	variables
flayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
://2layer7/kernel
:/2layer7/bias
.
30
41"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?

glayers
hlayer_regularization_losses
imetrics
5trainable_variables
jnon_trainable_variables
6regularization_losses
7	variables
klayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:/2layer8/kernel
:2layer8/bias
.
90
:1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?

llayers
mlayer_regularization_losses
nmetrics
;trainable_variables
onon_trainable_variables
<regularization_losses
=	variables
player_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
'
q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	rtotal
	scount
t	variables
u	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
r0
s1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
$:"//2Adam/layer1/kernel/m
:/2Adam/layer1/bias/m
$:"/d2Adam/layer2/kernel/m
:d2Adam/layer2/bias/m
$:"dd2Adam/layer3/kernel/m
:d2Adam/layer3/bias/m
$:"d/2Adam/layer4/kernel/m
:/2Adam/layer4/bias/m
$:"//2Adam/layer5/kernel/m
:/2Adam/layer5/bias/m
$:"//2Adam/layer6/kernel/m
:/2Adam/layer6/bias/m
$:"//2Adam/layer7/kernel/m
:/2Adam/layer7/bias/m
$:"/2Adam/layer8/kernel/m
:2Adam/layer8/bias/m
$:"//2Adam/layer1/kernel/v
:/2Adam/layer1/bias/v
$:"/d2Adam/layer2/kernel/v
:d2Adam/layer2/bias/v
$:"dd2Adam/layer3/kernel/v
:d2Adam/layer3/bias/v
$:"d/2Adam/layer4/kernel/v
:/2Adam/layer4/bias/v
$:"//2Adam/layer5/kernel/v
:/2Adam/layer5/bias/v
$:"//2Adam/layer6/kernel/v
:/2Adam/layer6/bias/v
$:"//2Adam/layer7/kernel/v
:/2Adam/layer7/bias/v
$:"/2Adam/layer8/kernel/v
:2Adam/layer8/bias/v
?2?
H__inference_sequential_8_layer_call_and_return_conditional_losses_331422
H__inference_sequential_8_layer_call_and_return_conditional_losses_331529
H__inference_sequential_8_layer_call_and_return_conditional_losses_331130
H__inference_sequential_8_layer_call_and_return_conditional_losses_331222?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_8_layer_call_fn_330755
-__inference_sequential_8_layer_call_fn_331566
-__inference_sequential_8_layer_call_fn_331603
-__inference_sequential_8_layer_call_fn_331038?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_330481input_9"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer1_layer_call_and_return_conditional_losses_331626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer1_layer_call_fn_331635?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer2_layer_call_and_return_conditional_losses_331658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer2_layer_call_fn_331667?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer3_layer_call_and_return_conditional_losses_331690?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer3_layer_call_fn_331699?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer4_layer_call_and_return_conditional_losses_331722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer4_layer_call_fn_331731?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer5_layer_call_and_return_conditional_losses_331754?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer5_layer_call_fn_331763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer6_layer_call_and_return_conditional_losses_331786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer6_layer_call_fn_331795?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer7_layer_call_and_return_conditional_losses_331818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer7_layer_call_fn_331827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer8_layer_call_and_return_conditional_losses_331849?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer8_layer_call_fn_331858?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_331869?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_331880?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_331891?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_331902?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_331913?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_331924?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_6_331935?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_7_331946?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
$__inference_signature_wrapper_331315input_9"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_330481u!"'(-.349:0?-
&?#
!?
input_9?????????/
? "/?,
*
layer8 ?
layer8??????????
B__inference_layer1_layer_call_and_return_conditional_losses_331626\/?,
%?"
 ?
inputs?????????/
? "%?"
?
0?????????/
? z
'__inference_layer1_layer_call_fn_331635O/?,
%?"
 ?
inputs?????????/
? "??????????/?
B__inference_layer2_layer_call_and_return_conditional_losses_331658\/?,
%?"
 ?
inputs?????????/
? "%?"
?
0?????????d
? z
'__inference_layer2_layer_call_fn_331667O/?,
%?"
 ?
inputs?????????/
? "??????????d?
B__inference_layer3_layer_call_and_return_conditional_losses_331690\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? z
'__inference_layer3_layer_call_fn_331699O/?,
%?"
 ?
inputs?????????d
? "??????????d?
B__inference_layer4_layer_call_and_return_conditional_losses_331722\!"/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????/
? z
'__inference_layer4_layer_call_fn_331731O!"/?,
%?"
 ?
inputs?????????d
? "??????????/?
B__inference_layer5_layer_call_and_return_conditional_losses_331754\'(/?,
%?"
 ?
inputs?????????/
? "%?"
?
0?????????/
? z
'__inference_layer5_layer_call_fn_331763O'(/?,
%?"
 ?
inputs?????????/
? "??????????/?
B__inference_layer6_layer_call_and_return_conditional_losses_331786\-./?,
%?"
 ?
inputs?????????/
? "%?"
?
0?????????/
? z
'__inference_layer6_layer_call_fn_331795O-./?,
%?"
 ?
inputs?????????/
? "??????????/?
B__inference_layer7_layer_call_and_return_conditional_losses_331818\34/?,
%?"
 ?
inputs?????????/
? "%?"
?
0?????????/
? z
'__inference_layer7_layer_call_fn_331827O34/?,
%?"
 ?
inputs?????????/
? "??????????/?
B__inference_layer8_layer_call_and_return_conditional_losses_331849\9:/?,
%?"
 ?
inputs?????????/
? "%?"
?
0?????????
? z
'__inference_layer8_layer_call_fn_331858O9:/?,
%?"
 ?
inputs?????????/
? "??????????;
__inference_loss_fn_0_331869?

? 
? "? ;
__inference_loss_fn_1_331880?

? 
? "? ;
__inference_loss_fn_2_331891?

? 
? "? ;
__inference_loss_fn_3_331902!?

? 
? "? ;
__inference_loss_fn_4_331913'?

? 
? "? ;
__inference_loss_fn_5_331924-?

? 
? "? ;
__inference_loss_fn_6_3319353?

? 
? "? ;
__inference_loss_fn_7_3319469?

? 
? "? ?
H__inference_sequential_8_layer_call_and_return_conditional_losses_331130s!"'(-.349:8?5
.?+
!?
input_9?????????/
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_8_layer_call_and_return_conditional_losses_331222s!"'(-.349:8?5
.?+
!?
input_9?????????/
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_8_layer_call_and_return_conditional_losses_331422r!"'(-.349:7?4
-?*
 ?
inputs?????????/
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_8_layer_call_and_return_conditional_losses_331529r!"'(-.349:7?4
-?*
 ?
inputs?????????/
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_8_layer_call_fn_330755f!"'(-.349:8?5
.?+
!?
input_9?????????/
p 

 
? "???????????
-__inference_sequential_8_layer_call_fn_331038f!"'(-.349:8?5
.?+
!?
input_9?????????/
p

 
? "???????????
-__inference_sequential_8_layer_call_fn_331566e!"'(-.349:7?4
-?*
 ?
inputs?????????/
p 

 
? "???????????
-__inference_sequential_8_layer_call_fn_331603e!"'(-.349:7?4
-?*
 ?
inputs?????????/
p

 
? "???????????
$__inference_signature_wrapper_331315?!"'(-.349:;?8
? 
1?.
,
input_9!?
input_9?????????/"/?,
*
layer8 ?
layer8?????????