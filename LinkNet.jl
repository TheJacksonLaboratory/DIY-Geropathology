using Flux
using Flux:@functor
using Metalhead
using BSON
using PyCall
using Distributions: Normal
using DataFrames
using CSV
using JLD2
using Conda
using FileIO
using ImageFiltering
using ImageMagick
using Images
using Interpolations
using ONNX
using Plots
using PyCall
using StatsBase
using StatsKit
using Random

function crop(x)
    x=x[2:end,2:end,:,:]
end
function lrelu(x)
    leakyrelu.(x,0.2f0)
end

function _random_normal(shape...)
  return Float32.(rand(Normal(0.f0,0.02f0),shape...))
end

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)
function squeeze(x)
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # For the case BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)...,1)
    end
end

function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2),
      BatchNorm(out_ch),
      x->squeeze(x))
end

struct UNetUpBlock
  upsample
end

@functor UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (3, 3), p = 0.5f0) =
    UNetUpBlock(Chain(lrelu,
           ConvTranspose((2, 2), in_chs=>out_chs,
            stride=(2, 2);init=_random_normal),
        BatchNorm(out_chs),
        Dropout(p)))

function (u::UNetUpBlock)(x, bridge)
  x = u.upsample(x)
  if isempty(bridge)
      return x
  else
  	  return cat(x, bridge, dims = 3)
  end
end

struct LinkNetUpBlock
  upsample
end
@functor LinkNetUpBlock

LinkNetUpBlock(in_chs::Int, out_chs::Int; kernel = (3, 3), p = 0.5f0) =
    LinkNetUpBlock(Chain(lrelu,
            ConvTranspose((2, 2), in_chs=>out_chs,
            stride=(2, 2);init=_random_normal),
        BatchNorm(out_chs), #BatchNormWrap(out_chs),
    Dropout(p)))

function (l::LinkNetUpBlock)(x, bridge)
  x = l.upsample(x)
  if isempty(bridge)
        return x
  else
       return x.+bridge
  end
end



struct Linknet
  reslayers
  up_blocks
end

@functor Linknet

function Linknet(channels::Int = 1, labels::Int = channels)
  reslayers=ResNet().layers

  up_blocks = Chain(LinkNetUpBlock(2048, 1024),
        LinkNetUpBlock(1024, 512),
        LinkNetUpBlock(512, 256),
        LinkNetUpBlock(256, 64,p = 0.0f0),
        Chain(lrelu,
        ConvTranspose((3, 3), 64=>32,stride=(2,2);init=_random_normal),
        BatchNorm(32), #BatchNormWrap(32),
        lrelu,
        Conv((1, 1), 32=>labels;init=_random_normal),
        crop))
  Linknet(reslayers, up_blocks)
end

function (l::Linknet)(x::AbstractArray)
  op = l.reslayers[1:2](x)

  x1 = l.reslayers[3:5](op)
  x2 = l.reslayers[6:9](x1)
  x3 = l.reslayers[10:15](x2)
  x4 = l.reslayers[16](x3)

  up_x1 = l.up_blocks[1](x4, x3)
  up_x2 = l.up_blocks[2](up_x1, x2)
  up_x3 = l.up_blocks[3](up_x2, x1)
  up_x5 = l.up_blocks[4](up_x3,[])
  up_x6 = tanh.(l.up_blocks[end](up_x5))
  σ.(up_x6)
end
