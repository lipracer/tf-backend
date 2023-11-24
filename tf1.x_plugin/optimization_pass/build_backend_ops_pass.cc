
#include <deque>
#include <unordered_map>

#include "../kernels/backend_ops.h"
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/tf1.x_plugin/include/adaptor.h"
#include "tensorflow/compiler/jit/tf1.x_plugin/include/op_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"
#include "white_list.h"

namespace tensorflow
{

namespace
{

class NodeReplacer
{
public:
    struct NodeWrap
    {
        NodeWrap(Node* o, Node* n) : old_node(o), new_node(n) {}
        Node* old_node;
        Node* new_node;
    };

    template <typename IterT>
    explicit NodeReplacer(Graph* graph, IterT first, IterT last, const std::string& requested_device_name,
                          const std::string& assigned_device_name)
        : graph_(graph),
          nodes_(first, last),
          requested_device_name_(requested_device_name),
          assigned_device_name_(assigned_device_name)
    {
    }

    void do_replace();

    ~NodeReplacer();

    bool isArgument(Node*);
    bool isReturn(Node*);
    bool isD2H(Node* node);

    void buildH2DNode(Node* node);
    void buildD2HNode(Node* node);

    void buildNormalNode(Node* node);

    bool hsaBackendImplement(Node* node);

    void set_device(Node* node)
    {
        VLOG(0) << "set_device:" << node->DebugString();
        "/job:localhost/replica:0/task:0/device:CPU:0";
        // node->set_requested_device(requested_device_name_);
        // node->set_assigned_device_name(assigned_device_name_);
        // node->set_requested_device("/job:localhost/replica:0/task:0/device:CPU:0");
        // node->set_assigned_device_name("/job:localhost/replica:0/task:0/device:CPU:0");
    }

    static size_t node_index()
    {
        static size_t index = 0;
        return index++;
    }

    static NodeWrap find_new_node_or_die(std::unordered_map<Node*, NodeWrap>& map, Node* node)
    {
        auto iter = map.find(node);
        if (iter == map.end())
        {
            VLOG(0) << "node:" << node << "\n" << node->DebugString();
            for (auto it : map)
            {
                VLOG(0) << "map:" << it.first << "\n" << it.first->DebugString();
            }
        }

        CHECK(iter != map.end());
        return iter->second;
    }

private:
    Graph* graph_;
    std::vector<Node*> nodes_;
    std::unordered_map<Node*, NodeWrap> node_map_;
    std::vector<Node*> remove_nodes_;
    std::vector<const Edge*> input_edges_;
    std::vector<const Edge*> output_edges_;

    std::string requested_device_name_;
    std::string assigned_device_name_;
};

NodeReplacer::~NodeReplacer()
{
    for (auto edge : input_edges_)
    {
        graph_->RemoveEdge(edge);
    }
    for (auto edge : output_edges_)
    {
        graph_->RemoveEdge(edge);
    }
    for (auto node : remove_nodes_)
    {
        graph_->RemoveNode(node);
    }
}

bool NodeReplacer::isArgument(Node* node)
{
    return "_Arg" == node->op_def().name();
}

bool NodeReplacer::isReturn(Node* node)
{
    return "_Retval" == node->op_def().name();
}

bool NodeReplacer::isD2H(Node* node)
{
    return "D2HOp" == node->op_def().name();
}

bool NodeReplacer::hsaBackendImplement(Node* node)
{
    return tfbe::lookupOpDef(tfbe::getOpLibs(), node->op_def().name().c_str());
}

void NodeReplacer::buildH2DNode(Node* node)
{
    VLOG(0) << node->DebugString();
    for (auto edge : node->out_edges())
    {
        input_edges_.push_back(edge);
    }
    Node* new_node = nullptr;

    NodeBuilder builder(/*name*/ "H2D_node_" + std::to_string(node_index()),
                        /*op name*/ "H2DOp");
    builder.Input(node);
    Status status = builder.Finalize(graph_, &new_node);
    if (!status.ok())
    {
        VLOG(0) << "backend_node error msg:" << status.error_message();
    }
    // set_device(node, new_node);
    VLOG(0) << "assigned_device_name:" << node->assigned_device_name();
    new_node->set_assigned_device_name(node->assigned_device_name());
    node_map_.insert(std::make_pair(node, NodeWrap(node, new_node)));
}

void NodeReplacer::buildD2HNode(Node* node)
{
    for (auto edge : node->in_edges())
    {
        output_edges_.push_back(edge);
    }
    auto input = *std::begin(node->in_nodes());
    if (!hsaBackendImplement(input))
    {
        graph_->AddEdge(find_new_node_or_die(node_map_, input).new_node, output_edges_[0]->src_output(), node,
                        output_edges_[0]->dst_input());
        return;
    }

    Node* new_node = nullptr;

    NodeBuilder builder(/*name*/ "D2H_node_" + std::to_string(node_index()),
                        /*op name*/ "D2HOp");
    for (auto input : node->in_nodes())
    {
        builder.Input(find_new_node_or_die(node_map_, input).new_node);
    }
    Status status = builder.Finalize(graph_, &new_node);
    if (!status.ok())
    {
        VLOG(0) << "backend_node error msg:" << status.error_message();
    }
    // set_device(new_node);
    VLOG(0) << "assigned_device_name:" << node->assigned_device_name();
    new_node->set_assigned_device_name(node->assigned_device_name());
    node_map_.insert(std::make_pair(node, NodeWrap(node, new_node)));

    graph_->AddEdge(new_node, output_edges_[0]->src_output(), output_edges_[0]->dst(), output_edges_[0]->dst_input());
}

void NodeReplacer::buildNormalNode(Node* node)
{
    if (!node->num_outputs())
    {
        return;
    }
    bool need_fallback = !hsaBackendImplement(node);
    VLOG(0) << "check hsaBackendImplement:" << !need_fallback;

    remove_nodes_.push_back(node);
    Node* new_node = nullptr;

    auto new_name = need_fallback ? node->op_def().name() : tfbe::OpNamePrefix + node->op_def().name();
    NodeBuilder builder(/*name*/ "backend_node_" + std::to_string(node_index()),
                        /*op name*/ new_name);

    auto insert_h2d = [graph_ = graph_](Node* input) -> Node*
    {
        Node* new_node = nullptr;
        NodeBuilder builder(/*name*/ "H2D_node_" + std::to_string(node_index()),
                            /*op name*/ "H2DOp");

        builder.Input(input);
        Status status = builder.Finalize(graph_, &new_node);
        if (!status.ok())
        {
            VLOG(0) << "backend_node error msg:" << status.error_message();
            return nullptr;
        }
        // set_device(new_node);
        VLOG(0) << "assigned_device_name:" << input->assigned_device_name();
        new_node->set_assigned_device_name(input->assigned_device_name());
        return new_node;
    };

    auto insert_d2h = [graph_ = graph_](Node* input) -> Node*
    {
        Node* new_node = nullptr;
        NodeBuilder builder(/*name*/ "D2H_node_" + std::to_string(node_index()),
                            /*op name*/ "D2HOp");

        builder.Input(input);
        Status status = builder.Finalize(graph_, &new_node);
        if (!status.ok())
        {
            VLOG(0) << "backend_node error msg:" << status.error_message();
            return nullptr;
        }
        // set_device(new_node);
        VLOG(0) << "assigned_device_name:" << input->assigned_device_name();
        new_node->set_assigned_device_name(input->assigned_device_name());
        return new_node;
    };

    for (auto input : node->in_nodes())
    {
        VLOG(0) << "visit normal input:" << input->op_def().name();
        Node* new_input = nullptr;
        VLOG(0) << "check hsaBackendImplement input:" << hsaBackendImplement(input);
        if (need_fallback)
        {
            if (!hsaBackendImplement(input) && !isArgument(input))
            {
                new_input = find_new_node_or_die(node_map_, input).new_node;
            }
            else if (isArgument(input))
            {
                new_input = input;
            }
            else
            {
                new_input = insert_d2h(find_new_node_or_die(node_map_, input).new_node);
            }
        }
        else
        {
            if (!hsaBackendImplement(input) && !isArgument(input))
            {
                new_input = insert_h2d(find_new_node_or_die(node_map_, input).new_node);
            }
            else
            {
                new_input = find_new_node_or_die(node_map_, input).new_node;
            }
        }
        builder.Input(new_input);
    }
    Status status = builder.Finalize(graph_, &new_node);
    if (!status.ok())
    {
        VLOG(0) << "backend_node error msg:" << status.error_message();
    }
    // set_device(new_node);
    VLOG(0) << "assigned_device_name:" << node->assigned_device_name();
    new_node->set_assigned_device_name(node->assigned_device_name());
    node_map_.insert(std::make_pair(node, NodeWrap(node, new_node)));
}

void NodeReplacer::do_replace()
{
    for (auto iter = nodes_.begin(); iter != nodes_.end(); ++iter)
    {
        VLOG(0) << "visit node op name:" << (*iter)->op_def().name();
        if (isArgument(*iter))
        {
            buildH2DNode(*iter);
        }
        else if (isReturn(*iter))
        {
            buildD2HNode(*iter);
        }
        else
        {
            buildNormalNode(*iter);
        }
    }
}

void replaceNodeWithBackendNode(Graph* graph, Node* node)
{
    VLOG(0) << "visit node op name:" << node->op_def().name();
    static std::set<std::string> white_list = {"AddV2", "Neg"};
    if (!white_list.count(node->op_def().name()))
    {
        return;
    }

    std::vector<Node*> in_nodes(std::begin(node->in_nodes()), std::end(node->in_nodes()));

    static size_t node_index = 0;
    Node* backend_node = nullptr;

    auto insertH2D = [graph](Node* arg) -> Node*
    {
        Node* H2DNode = nullptr;

        NodeBuilder builder(/*name*/ "H2D_node_" + std::to_string(node_index++),
                            /*op name*/ "H2DOp");
        builder.Input(arg);
        Status status = builder.Finalize(graph, &H2DNode);
        if (!status.ok())
        {
            VLOG(0) << "backend_node error msg:" << status.error_message();
        }
        H2DNode->set_requested_device(arg->requested_device());
        H2DNode->set_assigned_device_name(arg->assigned_device_name());
        return H2DNode;
    };

    auto insertD2H = [graph](Node* ret) -> Node*
    {
        Node* D2HNode = nullptr;

        NodeBuilder builder(/*name*/ "D2H_node_" + std::to_string(node_index++),
                            /*op name*/ "D2HOp");
        builder.Input(ret);
        Status status = builder.Finalize(graph, &D2HNode);
        if (!status.ok())
        {
            VLOG(0) << "backend_node error msg:" << status.error_message();
        }
        D2HNode->set_requested_device(ret->requested_device());
        D2HNode->set_assigned_device_name(ret->assigned_device_name());
        return D2HNode;
    };

    VLOG(0) << "replaceNodeWithBackendNode:" << node->name() << " vs op name:" << node->op_def().name()
            << " NodeDef:" << node->def().DebugString();

    NodeBuilder builder(/*name*/ "backend_node_" + std::to_string(node_index++),
                        /*op name*/ node->op_def().name() + "BackendOp");
    for (auto input : in_nodes)
    {
        if (input->op_def().name() == "_Arg")
        {
            input = insertH2D(input);
        }
        builder.Input(input);
    }

    Status status = builder.Finalize(graph, &backend_node);
    // remove origin node
    if (!status.ok())
    {
        VLOG(0) << "backend_node error msg:" << status.error_message();
    }
    else
    {
        const OpRegistrationData* op_reg_data = nullptr;
        OpRegistry::Global()->LookUp(node->op_def().name(), &op_reg_data);
        if (op_reg_data)
        {
            // TODO find shape infer func
        }
        // backend_node->AddAttr(kTfBackendEagerAttr, node->op_def().name());
        backend_node->set_requested_device(node->requested_device());
        backend_node->set_assigned_device_name(node->assigned_device_name());
        std::vector<const Edge*> edges;
        std::copy(std::begin(node->in_edges()), std::end(node->in_edges()), std::back_inserter(edges));
        for (auto edge : edges)
        {
            graph->RemoveEdge(edge);
        }
        edges.clear();
        std::copy(std::begin(node->out_edges()), std::end(node->out_edges()), std::back_inserter(edges));
        for (auto edge : edges)
        {
            auto new_node = backend_node;
            if ("_Retval" == edge->dst()->op_def().name())
            {
                new_node = insertD2H(backend_node);
            }
            graph->AddEdge(new_node, edge->src_output(), edge->dst(), edge->dst_input());
            graph->RemoveEdge(edge);
        }
        graph->RemoveNode(node);

        VLOG(0) << "replaceNodeWithBackendNode:" << backend_node->name()
                << " vs op name:" << backend_node->op_def().name() << " NodeDef:" << backend_node->def().DebugString();
    }
}
} // namespace

// Adds _XlaCompile and _XlaRun operations to the TF graph that compiles and
// executes (using XLA) TF function calls marked with "_XlaCompiledKernel".
class BuildBackendOpsPass : public GraphOptimizationPass
{
public:
    // If enable_lazy_compilation is not nullopt then *enable_lazy_compilation
    // overrides --tf_xla_enable_lazy_compilation flag in deciding whether lazy
    // compilation is enabled.
    explicit BuildBackendOpsPass(absl::optional<bool> enable_lazy_compilation = absl::nullopt)
        : enable_lazy_compilation_(enable_lazy_compilation)
    {
    }

    Status Run(const GraphOptimizationPassOptions& options) override;

private:
    absl::optional<bool> enable_lazy_compilation_;
};

Status BuildBackendOpsPass::Run(const GraphOptimizationPassOptions& options)
{
    Graph* graph = options.graph->get();
    std::deque<Node*> all_nodes;
    bool isLegalGraph = false;
    for (auto node : graph->nodes())
    {
        VLOG(0) << "collection node op name:" << node->op_def().name() << " node->IsArg():" << node->IsArg();
        if (node->IsArg())
        {
            isLegalGraph |= true;
            all_nodes.push_front(node);
            continue;
        }
        all_nodes.push_back(node);
    }
    VLOG(0) << "begin rewriter!";
    if (!isLegalGraph)
    {
        VLOG(0) << "ilegal graph===================";
        return Status::OK();
    }
    (void)FixupSourceAndSinkEdges(graph);
    // for (auto node : all_nodes)
    // {
    //     replaceNodeWithBackendNode(graph, node);
    // }

    {
        std::string requested_device_name = "/job:localhost/replica:0/task:0/device:CPU:0";
        std::string assigned_device_name = "/job:localhost/replica:0/task:0/device:CPU:0";
        NodeReplacer replacer(graph, std::begin(all_nodes), std::end(all_nodes), requested_device_name,
                              assigned_device_name);
        replacer.do_replace();
    }

    VLOG(0) << "graph:" << graph->ToGraphDefDebug().DebugString();

    return Status::OK();
}

} // namespace tensorflow

namespace tensorflow
{
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 60, BuildBackendOpsPass);
}