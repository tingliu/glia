#ifndef _glia_type_tree_hxx_
#define _glia_type_tree_hxx_

#include "type/object.hxx"
#include "util/container.hxx"
#include "util/text_io.hxx"
#include "alg/tree.hxx"

namespace glia {

class TreeNodeBase : public Object {
 public:
  typedef Object SuperObject;
  typedef TreeNodeBase Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;

  int self, parent;
  std::vector<int> children;

  TreeNodeBase () {}

  TreeNodeBase (int self, int parent) : self(self), parent(parent) {}

  TreeNodeBase (int self, int parent,
                std::initializer_list<int> const& children)
      : self(self), parent(parent) {
    if (children.size() != 0) {
      this->children.resize(children.size());
      std::copy(children.begin(), children.end(),
                this->children.begin());
    }
  }

  ~TreeNodeBase () override {}

  bool isLeaf () const { return children.empty(); }

  bool isRoot () const { return parent < 0; }

  friend std::ostream& operator<< (std::ostream& os, Self const& node){
    return os << node.self << " " << node.parent << " "
              << node.children;
  }

  friend std::istream& operator>> (std::istream& is, Self& node)
  { return is >> node.self >> node.parent >> node.children; }
};


template <typename T>
class TTreeNode : public TreeNodeBase {
 public:
  typedef TreeNodeBase Super;
  typedef TTreeNode<T> Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;
  typedef T Data;

  T data;

  TTreeNode () {}

  TTreeNode (int self, int parent) : Super(self, parent) {}

  TTreeNode (int self, int parent,
             std::initializer_list<int> const& children)
      : Super(self, parent, children) {}

  TTreeNode (int self, int parent,
             std::initializer_list<int> const& children,
             T const& data)
      : Super(self, parent, children), data(data) {}

  ~TTreeNode () override {}

  friend std::ostream& operator<< (std::ostream& os, Self const& node)
  { return os << static_cast<Super const&>(node) << " " << node.data; }

  friend std::istream& operator>> (std::istream& is, Self& node)
  { return is >> static_cast<Super&>(node) >> node.data; }
};


template <typename T>
class TTree : public Object, public std::vector<TTreeNode<T>> {
public:
  typedef Object SuperObject;
  typedef std::vector<TTreeNode<T>> Super;
  typedef TTree<T> Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;
  typedef TTreeNode<T> Node;

  ~TTree () override {}

  virtual int root () const { return Super::back().self; }

  template <typename Func> void
      traverseAncestors (int i, Func f) {
    for (i = Super::at(i).parent; i >= 0; i = Super::at(i).parent)
    { f(Super::at(i)); }
  }

  template <typename Func> void
      traverseAncestors (int i, Func f) const {
    for (i = Super::at(i).parent; i >= 0; i = Super::at(i).parent)
    { f(Super::at(i)); }
  }

  template <typename Func> void
      traverseDescendants (int i, Func f) {
    for (auto c: Super::at(i).children)
    { alg::bfs(*this, c, [c, f](Node& node){ f(node); }); }
  }

  template <typename Func> void
      traverseDescendants (int i, Func f) const {
    for (auto c: Super::at(i).children)
    { alg::bfs(*this, c, [c, f](Node const& node){ f(node); }); }
  }

  template <typename Func> void
      traverseSiblings (int i, Func f) {
    int pa = Super::at(i).parent;
    if (pa >= 0) {
      for (auto child: Super::at(pa).children)
      { if (child != i) { f(Super::at(child)); } }
    }
  }

  template <typename Func> void
      traverseSiblings (int i, Func f) const {
    int pa = Super::at(i).parent;
    if (pa >= 0) {
      for (auto child: Super::at(pa).children)
      { if (child != i) { f(Super::at(child)); } }
    }
  }

  template <typename Func> void
      traverseLeaves (int i, Func f) {
    if (Super::at(i).isLeaf()) {
      f(Super::at(i));
      return;
    }
    traverseDescendants(i, [&f](Node& node)
                        { if (node.isLeaf()) { f(node); } });
  }

  template <typename Func> void
      traverseLeaves (int i, Func f) const {
    if (Super::at(i).isLeaf()) {
      f(Super::at(i));
      return;
    }
    traverseDescendants(i, [&f](Node const& node)
                        { if (node.isLeaf()) { f(node); } });
  }

  virtual uint countLeaves (int i) const {
    uint ret = 0;
    traverseLeaves(i, [&ret](Node const& node){ ++ret; });
    return ret;
  }

  // Number of nodes in subtree
  virtual uint countNodes (int i) const {
    uint ret = 1;
    traverseDescendants(i, [&ret](Node const& node){ ++ret; });
    return ret;
  }

  virtual uint depth (int i) const {
    uint ret = 0;
    i = Super::at(i).parent;
    while (i >= 0) {
      ++ret;
      i = Super::at(i).parent;
    }
    return ret;
  }

  virtual uint height (int i) const {
    if (Super::at(i).isLeaf()) { return 0; }
    uint ret = 0;
    for (auto child: Super::at(i).children)
    { ret = std::max(ret, height(child)); }
    return ret + 1;
  }

  // f(subtreeNode, node): node data transfer function
  template <typename TSubtree, typename Func> void
      subtree (TSubtree& stree, int i, Func f) const {
    uint n = countNodes(i);
    stree.resize(n);
    std::unordered_map<int, int> imap;
    int j = n - 1;
    f(stree[j], Super::at(i));
    imap[i] = j--;
    traverseDescendants(i, [&j, &stree, &imap, &f](Node const& node)
                        {
                          f(stree[j], node);
                          imap[node.self] = j--;
                        });
    for (auto& node: stree) {
      node.self = imap.find(node.self)->second;
      node.parent = citerator(imap, node.parent, -1)->second;
      for (auto& child: node.children)
      { child = imap.find(child)->second; }
    }
  }
};

};

#endif
