// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include <fstream>

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
    using DeviceOpKernel<{0}>::DeviceOpKernel;
    void compute(tfbe::DeviceOpKernelContext* ctx)
    {{


    }
};
)";

constexpr char HeaderFiles[] = R"(

#include "kernel_context.h"
#include "logger/logger.h"
#include "op_registry.h"
#include "type/tensor.h"

)";

class CodeGenerator
{
public:
    static CodeGenerator& instance();

    void writeString(StringRef str)
    {
        os_ << str;
    }

    class CodeEmiter
    {
    public:
        CodeEmiter(CodeGenerator* generator);

        CodeEmiter& addInput(StringRef name, StringRef type);
        CodeEmiter& addTensorInput(StringRef name);

        CodeEmiter& addName(StringRef name);

        CodeEmiter& addAttr(StringRef key, StringRef value);
        CodeEmiter& addOutput(StringRef name);

        ~CodeEmiter();

        void emitClass();
        void emitRegistry();

    private:
        std::string buf_;
        llvm::raw_string_ostream os_;
        CodeGenerator* generator_;

        StringRef name_;
        SmallVector<std::pair<StringRef, StringRef>> inputs_;
        SmallVector<std::pair<StringRef, StringRef>> attrs_;
        StringRef output_;
    };

    CodeEmiter getEmiter();

    ~CodeGenerator()
    {
        os_.flush();
        std::string path = output_path;
        if (path.back() != '/')
        {
            path.push_back('/');
        }
        path += "op_registry.cpp";
        std::ofstream ofs(path);
        if (!ofs.is_open())
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

CodeGenerator::CodeEmiter CodeGenerator::getEmiter()
{
    return CodeEmiter(this);
}

CodeGenerator::CodeEmiter::CodeEmiter(CodeGenerator* generator) : os_(buf_), generator_(generator) {}

void CodeGenerator::CodeEmiter::emitClass()
{
    os_ << llvm::formatv(ClassTemplate, name_);
}

void CodeGenerator::CodeEmiter::emitRegistry()
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

CodeGenerator::CodeEmiter::~CodeEmiter()
{
    emitClass();
    emitRegistry();
    os_.flush();
    generator_->writeString(buf_);
}

CodeGenerator::CodeEmiter& CodeGenerator::CodeEmiter::addName(StringRef name)
{
    name_ = name;
    return *this;
}

CodeGenerator::CodeEmiter& CodeGenerator::CodeEmiter::addInput(StringRef name, StringRef type)
{
    inputs_.emplace_back(name, type);
    return *this;
}

CodeGenerator::CodeEmiter& CodeGenerator::CodeEmiter::addTensorInput(StringRef name)
{
    inputs_.emplace_back(name, "T");
    return *this;
}

CodeGenerator::CodeEmiter& CodeGenerator::CodeEmiter::addAttr(StringRef key, StringRef value)
{
    attrs_.emplace_back(key, value);
    return *this;
}

CodeGenerator::CodeEmiter& CodeGenerator::CodeEmiter::addOutput(StringRef name)
{
    output_ = name;
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
            auto emiter = CodeGenerator::instance().getEmiter();
            emiter.addName(funcDecl->getName());
            StringRef returnName = funcDecl->getDeclaredReturnType().getAsString();
            if (returnName == "Tensor")
            {
                emiter.addOutput(returnName);
            }
            for (const auto& param : llvm::make_range(funcDecl->param_begin(), funcDecl->param_end()))
            {
                auto typeName = param->getType().split().asPair().first->getPointeeCXXRecordDecl()->getName();
                if (typeName == "Tensor")
                {
                    emiter.addInput(param->getName(), typeName);
                }
                else
                {
                    emiter.addAttr(param->getName(), typeName);
                }
            }
        }
    }
};

// auto CxxMemberCallExprM = cxxDependentScopeMemberExpr().bind("memberCall");

int main(int argc, const char** argv)
{
    (void)CodeGenerator::instance();
    CodeGenerator::instance().writeString("");
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
