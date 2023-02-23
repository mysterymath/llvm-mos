//===- XO65.h - XO65 object file implementation -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the XO65ObjectFile class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_XO65_H
#define LLVM_OBJECT_XO65_H

#include "llvm/Object/ObjectFile.h"

namespace llvm {
namespace object {

class XO65ObjectFile final : public ObjectFile {
public:
  static Expected<std::unique_ptr<XO65ObjectFile>>
  create(MemoryBufferRef Object);

  void moveSymbolNext(DataRefImpl &Symb) const override;
  Expected<uint32_t> getSymbolFlags(DataRefImpl Symb) const override;

  basic_symbol_iterator symbol_begin() const override;
  basic_symbol_iterator symbol_end() const override;

  bool is64Bit() const override;

  section_iterator section_begin() const override;
  section_iterator section_end() const override;

  uint8_t getBytesInAddress() const override;

  StringRef getFileFormatName() const override;
  Triple::ArchType getArch() const override;
  Expected<SubtargetFeatures> getFeatures() const override;

  bool isRelocatableObject() const override;

private:
  XO65ObjectFile(MemoryBufferRef Object);

  Error initialize();

  Expected<StringRef> getSymbolName(DataRefImpl Symb) const override;
  Expected<uint64_t> getSymbolAddress(DataRefImpl Symb) const override;
  uint64_t getSymbolValueImpl(DataRefImpl Symb) const override;
  uint64_t getCommonSymbolSizeImpl(DataRefImpl Symb) const override;
  Expected<SymbolRef::Type> getSymbolType(DataRefImpl Symb) const override;
  Expected<section_iterator> getSymbolSection(DataRefImpl Symb) const override;

  void moveSectionNext(DataRefImpl &Sec) const override;
  Expected<StringRef> getSectionName(DataRefImpl Sec) const override;
  uint64_t getSectionAddress(DataRefImpl Sec) const override;
  uint64_t getSectionIndex(DataRefImpl Sec) const override;
  uint64_t getSectionSize(DataRefImpl Sec) const override;
  Expected<ArrayRef<uint8_t>>
  getSectionContents(DataRefImpl Sec) const override;
  uint64_t getSectionAlignment(DataRefImpl Sec) const override;
  bool isSectionCompressed(DataRefImpl Sec) const override;
  bool isSectionText(DataRefImpl Sec) const override;
  bool isSectionData(DataRefImpl Sec) const override;
  bool isSectionBSS(DataRefImpl Sec) const override;
  bool isSectionVirtual(DataRefImpl Sec) const override;
  relocation_iterator section_rel_begin(DataRefImpl Sec) const override;
  relocation_iterator section_rel_end(DataRefImpl Sec) const override;

  void moveRelocationNext(DataRefImpl &Rel) const override;
  uint64_t getRelocationOffset(DataRefImpl Rel) const override;
  symbol_iterator getRelocationSymbol(DataRefImpl Rel) const override;
  uint64_t getRelocationType(DataRefImpl Rel) const override;
  void getRelocationTypeName(DataRefImpl Rel,
                             SmallVectorImpl<char> &Result) const override;
};

} // namespace object
} // namespace llvm

#endif // not LLVM_OBJECT_XO65_H
