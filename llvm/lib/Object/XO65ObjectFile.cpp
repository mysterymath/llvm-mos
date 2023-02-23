//===- XO65ObjectFile.cpp - COFF object file implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the COFFObjectFile class.
///
//===----------------------------------------------------------------------===//

#include "llvm/Object/XO65.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace object;

Expected<std::unique_ptr<XO65ObjectFile>>
ObjectFile::createXO65ObjectFile(MemoryBufferRef Object) {
  return XO65ObjectFile::create(Object);
}

Expected<std::unique_ptr<XO65ObjectFile>>
XO65ObjectFile::create(MemoryBufferRef Object) {
  std::unique_ptr<XO65ObjectFile> Obj(new XO65ObjectFile(std::move(Object)));
  if (Error E = Obj->initialize())
    return std::move(E);
  return std::move(Obj);
}

XO65ObjectFile::XO65ObjectFile(MemoryBufferRef Object)
    : ObjectFile(Binary::ID_XO65, Object) {}

Error XO65ObjectFile::initialize() { llvm_unreachable("Not yet implemented."); }

void XO65ObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  llvm_unreachable("Not yet implemented.");
}
