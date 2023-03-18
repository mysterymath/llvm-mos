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

#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/XO65.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace object;

Expected<std::unique_ptr<XO65ObjectFile>>
XO65ObjectFile::create(MemoryBufferRef Object) {
  std::unique_ptr<XO65ObjectFile> Obj(new XO65ObjectFile(std::move(Object)));
  if (Error E = Obj->initialize())
    return std::move(E);
  return std::move(Obj);
}

void XO65ObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  llvm_unreachable("Not yet implemented.");
}

Expected<uint32_t> XO65ObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  llvm_unreachable("Not yet implemented.");
}

basic_symbol_iterator XO65ObjectFile::symbol_begin() const {
  llvm_unreachable("Not yet implemented.");
}
basic_symbol_iterator XO65ObjectFile::symbol_end() const {
  llvm_unreachable("Not yet implemented.");
}

bool XO65ObjectFile::is64Bit() const {
  llvm_unreachable("Not yet implemented.");
}

section_iterator XO65ObjectFile::section_begin() const {
  llvm_unreachable("Not yet implemented.");
}
section_iterator XO65ObjectFile::section_end() const {
  llvm_unreachable("Not yet implemented.");
}

uint8_t XO65ObjectFile::getBytesInAddress() const {
  llvm_unreachable("Not yet implemented.");
}

StringRef XO65ObjectFile::getFileFormatName() const {
  llvm_unreachable("Not yet implemented.");
}
Triple::ArchType XO65ObjectFile::getArch() const {
  llvm_unreachable("Not yet implemented.");
}
Expected<SubtargetFeatures> XO65ObjectFile::getFeatures() const {
  llvm_unreachable("Not yet implemented.");
}

bool XO65ObjectFile::isRelocatableObject() const {
  llvm_unreachable("Not yet implemented.");
}

XO65ObjectFile::XO65ObjectFile(MemoryBufferRef Object)
    : ObjectFile(Binary::ID_XO65, Object) {}

Error XO65ObjectFile::initialize() { return Error::success(); }

Expected<StringRef> XO65ObjectFile::getSymbolName(DataRefImpl Symb) const {
  llvm_unreachable("Not yet implemented.");
}
Expected<uint64_t> XO65ObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  llvm_unreachable("Not yet implemented.");
}
uint64_t XO65ObjectFile::getSymbolValueImpl(DataRefImpl Symb) const {
  llvm_unreachable("Not yet implemented.");
}
uint64_t XO65ObjectFile::getCommonSymbolSizeImpl(DataRefImpl Symb) const {
  llvm_unreachable("Not yet implemented.");
}
Expected<SymbolRef::Type>
XO65ObjectFile::getSymbolType(DataRefImpl Symb) const {
  llvm_unreachable("Not yet implemented.");
}
Expected<section_iterator>
XO65ObjectFile::getSymbolSection(DataRefImpl Symb) const {
  llvm_unreachable("Not yet implemented.");
}

void XO65ObjectFile::moveSectionNext(DataRefImpl &Sec) const {
  llvm_unreachable("Not yet implemented.");
}
Expected<StringRef> XO65ObjectFile::getSectionName(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
uint64_t XO65ObjectFile::getSectionAddress(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
uint64_t XO65ObjectFile::getSectionIndex(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
uint64_t XO65ObjectFile::getSectionSize(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
Expected<ArrayRef<uint8_t>>
XO65ObjectFile::getSectionContents(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
uint64_t XO65ObjectFile::getSectionAlignment(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
bool XO65ObjectFile::isSectionCompressed(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
bool XO65ObjectFile::isSectionText(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
bool XO65ObjectFile::isSectionData(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
bool XO65ObjectFile::isSectionBSS(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
bool XO65ObjectFile::isSectionVirtual(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
relocation_iterator XO65ObjectFile::section_rel_begin(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}
relocation_iterator XO65ObjectFile::section_rel_end(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented.");
}

void XO65ObjectFile::moveRelocationNext(DataRefImpl &Rel) const {
  llvm_unreachable("Not yet implemented.");
}
uint64_t XO65ObjectFile::getRelocationOffset(DataRefImpl Rel) const {
  llvm_unreachable("Not yet implemented.");
}
symbol_iterator XO65ObjectFile::getRelocationSymbol(DataRefImpl Rel) const {
  llvm_unreachable("Not yet implemented.");
}
uint64_t XO65ObjectFile::getRelocationType(DataRefImpl Rel) const {
  llvm_unreachable("Not yet implemented.");
}
void XO65ObjectFile::getRelocationTypeName(
    DataRefImpl Rel, SmallVectorImpl<char> &Result) const {
  llvm_unreachable("Not yet implemented.");
}

Expected<std::unique_ptr<XO65ObjectFile>>
ObjectFile::createXO65ObjectFile(MemoryBufferRef Object) {
  return XO65ObjectFile::create(Object);
}
