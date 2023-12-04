// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang::tooling;
using namespace llvm;

// CHECK glib abi
static_assert(_GLIBCXX_USE_CXX11_ABI == 1, "");

static llvm::cl::OptionCategory MyToolCategory("my-tool options");
static cl::opt<std::string> output_path("output-path", cl::desc("rcodegen output path]"), cl::value_desc("a full path"),
                                        cl::ValueRequired, cl::NotHidden, cl::cat(MyToolCategory));

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.

using namespace clang;
using namespace clang::ast_matchers;

constexpr char ClassTemplate[] = R"(
class {0} : public tfbe::DeviceOpKernel<{0}>
{
public:
    {0}(tfbe::CompilerContext* ctx) : tfbe::DeviceOpKernel<{0}>(ctx) {{
{1}
    }
    void compute(tfbe::DeviceOpKernelContext* ctx)
    {{
{2}
    }
private:
    struct CAttr {{
{3}
    } attrs_;
};
)";

constexpr char HeaderFiles[] = R"(

#include "adt/tensor.h"
#include "kernel_context.h"
#include "logger/logger.h"
#include "native_ops.h"
#include "op_registry.h"

// codegen file don't edit

)";

class CodeGenerator
{
public:
    static CodeGenerator& instance();

    class CodeEmitter
    {
    public:
        CodeEmitter(llvm::raw_ostream& os);

        CodeEmitter& addInput(StringRef name, StringRef type);
        CodeEmitter& addTensorInput(StringRef name);

        CodeEmitter& addName(StringRef name);

        CodeEmitter& addAttr(StringRef key, StringRef value);
        CodeEmitter& addOutput(StringRef name);

        ~CodeEmitter();

        void emitClass();
        void emitRegistry();

    private:
        llvm::raw_ostream& os_;

        std::string name_;
        SmallVector<std::pair<std::string, std::string>> inputs_;
        SmallVector<std::pair<std::string, std::string>> attrs_;
        std::string output_;
    };

    CodeEmitter getEmitter();

    ~CodeGenerator()
    {
        os_.flush();
        std::string path = output_path;
        if (path.back() != '/')
        {
            path.push_back('/');
        }
        path += "op_registry.cpp";

        std::error_code EC;
        llvm::raw_fd_ostream ofs(path, EC);
        if (EC.value() != 0)
        {
            llvm_unreachable("can't open file!");
        }
        ofs << buf_;
        ofs.flush();
        ofs.close();
    }

private:
    CodeGenerator() : os_(buf_)
    {
        os_ << HeaderFiles;
    }

    std::string buf_;
    llvm::raw_string_ostream os_;
};

CodeGenerator& CodeGenerator::instance()
{
    static CodeGenerator generator;
    return generator;
}

CodeGenerator::CodeEmitter CodeGenerator::getEmitter()
{
    return CodeEmitter(os_);
}

class ScopeEmitter
{
public:
    ScopeEmitter(size_t indent) : indent_(indent), os_(buf_) {}

    template <typename T>
    friend ScopeEmitter& operator<<(ScopeEmitter&, T&& t);

    void newLine()
    {
        os_ << "\n";
        for (size_t i = 0; i < indent_; ++i)
        {
            os_ << "\t";
        }
    }

    std::string& str()
    {
        os_.flush();
        return buf_;
    }

private:
    size_t indent_;
    std::string buf_;
    llvm::raw_string_ostream os_;
};

template <typename T>
ScopeEmitter& operator<<(ScopeEmitter& e, T&& t)
{
    e.os_ << std::forward<T>(t);
    return e;
}

CodeGenerator::CodeEmitter::CodeEmitter(llvm::raw_ostream& os) : os_(os) {}

void CodeGenerator::CodeEmitter::emitClass()
{

    ScopeEmitter initEmitter(2);
    initEmitter.newLine();
    initEmitter << "attrs_ = {";
    llvm::interleaveComma(attrs_, initEmitter,
                          [&](auto& str)
                          {
                              constexpr char fmt[] = R"(.{0} = ctx->getAttr<{1}>("{0}"))";
                              initEmitter << llvm::formatv(fmt, str.first, str.second);
                          });
    initEmitter << "};";

    ScopeEmitter callEmitter(2);
    std::vector<std::string> params;
    for (size_t i = 0; i < inputs_.size(); ++i)
    {
        params.push_back(std::string("ctx->input(") + std::to_string(i) + ")");
    }
    for (auto& attr : attrs_)
    {
        params.push_back(std::string("attrs_.") + attr.first);
    }
    callEmitter.newLine();
    callEmitter << "auto result = " << "tfbe::autogen::" << name_.substr(1) << "(";
    llvm::interleaveComma(params, callEmitter, [&](auto& str) { callEmitter << str; });
    callEmitter << ");";

    callEmitter.newLine();
    callEmitter << "ctx->setOutput(0, result);";

    ScopeEmitter attrEmitter(2);
    for (auto& attr : attrs_)
    {
        attrEmitter.newLine();
        attrEmitter << llvm::formatv("{0} {1};", attr.second, attr.first);
    }
    os_ << llvm::formatv(ClassTemplate, name_, initEmitter.str(), callEmitter.str(), attrEmitter.str());
}

void CodeGenerator::CodeEmitter::emitRegistry()
{
    os_ << "\n";
    constexpr char RegistryStr[] = R"(REGISTER_KERNEL("{0}", {0}))";
    os_ << llvm::formatv(RegistryStr, name_);

    os_ << llvm::formatv(".Output(\"results : T\")");
    for (auto input : inputs_)
    {
        os_ << llvm::formatv(".Input(\"{0} : {1}\")", input.first, input.second);
    }
    for (auto& attr : attrs_)
    {
        os_ << llvm::formatv(".Attr(\"{0} : {1}\")", attr.first, attr.second);
    }
    os_ << ";\n";
}

CodeGenerator::CodeEmitter::~CodeEmitter()
{
    emitClass();
    emitRegistry();
    os_.flush();
}

CodeGenerator::CodeEmitter& CodeGenerator::CodeEmitter::addName(StringRef name)
{
    name_ = "S" + name.str();
    return *this;
}

CodeGenerator::CodeEmitter& CodeGenerator::CodeEmitter::addInput(StringRef name, StringRef type)
{
    inputs_.emplace_back(name.str(), type.str());
    return *this;
}

CodeGenerator::CodeEmitter& CodeGenerator::CodeEmitter::addTensorInput(StringRef name)
{
    inputs_.emplace_back(name.str(), "T");
    return *this;
}

CodeGenerator::CodeEmitter& CodeGenerator::CodeEmitter::addAttr(StringRef key, StringRef value)
{
    attrs_.emplace_back(key.str(), value.str());
    return *this;
}

CodeGenerator::CodeEmitter& CodeGenerator::CodeEmitter::addOutput(StringRef name)
{
    output_ = name.str();
    return *this;
}

class RegistryCodegen : public MatchFinder::MatchCallback
{
public:
    void run(const MatchFinder::MatchResult& Result) override
    {
        if (const FunctionDecl* funcDecl = Result.Nodes.getNodeAs<FunctionDecl>("function"))
        {
            const DeclContext* DeclCtx = funcDecl->getDeclContext();

            const auto* Namespace = dyn_cast<NamespaceDecl>(DeclCtx);
            if (!Namespace)
            {
                return;
            }
            if (Namespace->getNameAsString() != "autogen")
            {
                return;
            }
            auto emitter = CodeGenerator::instance().getEmitter();
            emitter.addName(funcDecl->getName());
            StringRef returnName = funcDecl->getDeclaredReturnType().getAsString();
            if (returnName == "Tensor")
            {
                emitter.addOutput(returnName);
            }
            auto paramList = funcDecl->parameters();
            for (const auto& param : paramList)
            {
                std::string typeName;
                auto nonRefType = param->getType().getNonReferenceType();
                auto pair = param->getType().split();
                if (!nonRefType.isConstQualified())
                {
                    typeName = param->getType().getAsString();
                }
                else
                {
                    typeName = pair.asPair().first->getPointeeCXXRecordDecl()->getName();
                }

                if (typeName == "Tensor")
                {
                    emitter.addInput(param->getName(), "T");
                }
                else
                {
                    emitter.addAttr(param->getName(), typeName);
                }
            }
        }
    }
};

// auto CxxMemberCallExprM = cxxDependentScopeMemberExpr().bind("memberCall");

int main(int argc, const char** argv)
{
    (void)CodeGenerator::instance();
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory, cl::Optional, nullptr);
    if (!ExpectedParser)
    {
        // Fail gracefully for unsupported options.
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    CommonOptionsParser& OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    MatchFinder Finder;

    auto funcDeclMatcher = functionDecl().bind("function");

    RegistryCodegen codegen;
    Finder.addMatcher(funcDeclMatcher, &codegen);

    int ExitCode = Tool.run(newFrontendActionFactory(&Finder).get());
    LangOptions DefaultLangOptions;
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
    TextDiagnosticPrinter DiagnosticPrinter(errs(), &*DiagOpts);
    DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
                                  &DiagnosticPrinter, false);

    return ExitCode;
}
